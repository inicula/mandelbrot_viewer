#define OLC_PGE_APPLICATION
#include "olcPixelGameEngine.h"

#include <condition_variable>
#include <atomic>
#include <complex>
#include <array>
#include <cstdlib>
#include <immintrin.h>

std::tuple<int, int, int> get_rgb(int n, int iter_max)
{
  double t = (double)n / (double)iter_max;
  int r = (int)(9 * (1 - t) * t * t * t * 255);
  int g = (int)(15 * (1 - t) * (1 - t) * t * t * 255);
  int b = (int)(8.5 * (1 - t) * (1 - t) * (1 - t) * t * 255);
  return std::tuple<int, int, int>(r, g, b);
}

template<typename T>
void prt(T&& v1, T&& v2)
{
  std::cout << v1.x << " " << v1.y << " -- " << v2.x << " " << v2.y << "\n";
}

template<typename T>
T operator+(const T& left, const T& right)
{
  return {left.x + right.x, left.y + right.y};
}

class Fractal_viewer : public olc::PixelGameEngine
{
public:
  using uint = std::uint32_t;
  using Vector2d = olc::vd2d;
  using Vector2ui = olc::vu2d;
  Fractal_viewer()
  {
    sAppName = "Fractal Viewer";
  }

public:
  bool OnUserCreate() override
  {
    // aligned_alloc: function for aligned allocation on Linux
    // https://man7.org/linux/man-pages/man3/aligned_alloc.3.html
    pixel_iterations = static_cast<uint*>(aligned_alloc(64, screen_size * sizeof(uint)));
    return true;
  }

  bool OnUserDestroy() override
  {
    free(pixel_iterations);
    return true;
  }

  void iterate_vanilla(const Vector2ui& screen_top_left, const Vector2ui& screen_bottom_right,
                       const Vector2d& fractal_top_left, const double x_scale,
                       const double y_scale, const uint max_iterations)
  {
    uint y_offset = screen_top_left.y * screen_width;
    double y_pos = screen_top_left.y * y_scale + fractal_top_left.y;

    for(uint y = screen_top_left.y; y < screen_bottom_right.y; y++)
    {
      double x_pos = fractal_top_left.x;
      double c_imag = y_pos;
      for(uint x = screen_top_left.x; x < screen_bottom_right.x; x++)
      {
        double c_real = x_pos;
        double z_real = 0;
        double z_imag = 0;

        uint n = 0;
        while((z_real * z_real + z_imag * z_imag) < 4.0 && n < max_iterations)
        {
          double real = z_real * z_real - z_imag * z_imag + c_real;
          double imag = z_real * z_imag * 2.0 + c_imag;
          z_real = real;
          z_imag = imag;
          ++n;
        }

        pixel_iterations[y_offset + x] = n;
        x_pos += x_scale;
      }
      y_offset += screen_width;
      y_pos += y_scale;
    }
  }

  void iterate_simd(const Vector2ui& screen_top_left, const Vector2ui& screen_bottom_right,
                    const Vector2d& fractal_top_left, const double x_scale,
                    const double y_scale, const uint max_iterations)
  {
    uint y_offset = screen_top_left.y * screen_width;
    double y_pos = screen_top_left.y * y_scale + fractal_top_left.y;

    const auto x_pos_start_4 = _mm256_add_pd(
        _mm256_mul_pd(_mm256_set1_pd(x_scale), _mm256_set_pd(0.0, 1.0, 2.0, 3.0)),
        _mm256_set1_pd(fractal_top_left.x));

    const auto max_iterations_4 = _mm256_set1_epi64x(max_iterations);
    const auto step_4 = _mm256_set1_pd(x_scale * 4.0);

    for(uint y = screen_top_left.y; y < screen_bottom_right.y; y++)
    {
      auto x_pos_4 = x_pos_start_4;
      auto c_imag_4 = _mm256_set1_pd(y_pos);
      for(uint x = screen_top_left.x; x < screen_bottom_right.x; x += 4)
      {
        auto c_real_4 = x_pos_4;
        auto z_real_4 = _mm256_setzero_pd();
        auto z_imag_4 = _mm256_setzero_pd();

        auto n_4 = _mm256_setzero_si256();
        auto condition = _mm256_and_si256(
            _mm256_castpd_si256(_mm256_cmp_pd(_mm256_add_pd(_mm256_mul_pd(z_real_4, z_real_4),
                                                            _mm256_mul_pd(z_imag_4, z_imag_4)),
                                              _mm256_set1_pd(4.0), _CMP_LT_OQ)),
            _mm256_cmpgt_epi64(max_iterations_4, n_4));
        while(_mm256_movemask_pd(_mm256_castsi256_pd(condition)) > 0)
        {
          auto z_real_4_squared = _mm256_mul_pd(z_real_4, z_real_4);
          auto z_imag_4_squared = _mm256_mul_pd(z_imag_4, z_imag_4);
          auto real_4 =
              _mm256_add_pd(_mm256_sub_pd(z_real_4_squared, z_imag_4_squared), c_real_4);
          auto imag_4 = _mm256_fmadd_pd(_mm256_mul_pd(z_real_4, z_imag_4), _mm256_set1_pd(2.0),
                                        c_imag_4);

          z_real_4 = real_4;
          z_imag_4 = imag_4;

          n_4 = _mm256_add_epi64(n_4, _mm256_and_si256(_mm256_set1_epi64x(1), condition));

          condition = _mm256_and_si256(_mm256_castpd_si256(_mm256_cmp_pd(
                                           _mm256_add_pd(z_real_4_squared, z_imag_4_squared),
                                           _mm256_set1_pd(4.0), _CMP_LT_OQ)),
                                       _mm256_cmpgt_epi64(max_iterations_4, n_4));
        }
        pixel_iterations[y_offset + x] = uint(n_4[3]);
        pixel_iterations[y_offset + x + 1] = uint(n_4[2]);
        pixel_iterations[y_offset + x + 2] = uint(n_4[1]);
        pixel_iterations[y_offset + x + 3] = uint(n_4[0]);
        x_pos_4 = _mm256_add_pd(x_pos_4, step_4);
      }
      y_offset += screen_width;
      y_pos += y_scale;
    }
  }

  template<typename Pred>
  void deploy_threads(Pred predicate, const Vector2ui& screen_top_left,
                      const Vector2ui& screen_bottom_right, const Vector2d& fractal_top_left,
                      const double x_scale, const double y_scale, const uint max_iterations)
  {
    uint screen_chunk_size = screen_height / n_threads;
    // std::cout << screen_chunk_size << "\n";

    std::vector<std::thread> threads(n_threads - 1);
    auto current_screen_pos = screen_top_left;
    for(uint i = 0; i < n_threads - 1; ++i)
    {
      auto next_screen_pos = current_screen_pos + Vector2ui{screen_width, screen_chunk_size};
      threads[i] = std::thread(predicate, *this, current_screen_pos, next_screen_pos,
                               fractal_top_left, x_scale, y_scale, max_iterations);
      current_screen_pos += {0, screen_chunk_size};
    }
    predicate(*this, current_screen_pos, screen_bottom_right, fractal_top_left, x_scale,
              y_scale, max_iterations);
    for(auto& thread : threads)
    {
      thread.join();
    }
  }

  template<typename Pred>
  void deploy_thread_pool(Pred predicate, const Vector2ui& screen_top_left,
                          const Vector2ui& screen_bottom_right,
                          const Vector2d& fractal_top_left, const double x_scale,
                          const double y_scale, const uint max_iterations)
  {
    uint screen_chunk_size = screen_height / n_threads;
    // std::cout << screen_chunk_size << "\n";
    auto current_screen_pos = screen_top_left;
    for(uint i = 0; i < n_threads - 1; ++i)
    {
      auto next_screen_pos = current_screen_pos + Vector2ui{screen_width, screen_chunk_size};
      thread_pool[i] = std::thread(predicate, *this, current_screen_pos, next_screen_pos,
                                   fractal_top_left, x_scale, y_scale, max_iterations);
      current_screen_pos += {0, screen_chunk_size};
    }
    predicate(*this, current_screen_pos, screen_bottom_right, fractal_top_left, x_scale,
              y_scale, max_iterations);
    for(uint i = 0; i < n_threads - 1; ++i)
    {
      thread_pool[i].join();
    }
  }

  bool OnUserUpdate(float fElapsedTime) override
  {

    // Get mouse location this frame
    Vector2d mouse_pos = {(double)GetMouseX(), (double)GetMouseY()};

    // Handle Pan & Zoom
    if(GetMouse(2).bPressed)
    {
      panning_pivot = mouse_pos;
    }

    if(GetMouse(2).bHeld)
    {
      camera_offset -= (mouse_pos - panning_pivot) / scale;
      panning_pivot = mouse_pos;
    }

    auto mouse_pos_before_zoom = screen_to_world(mouse_pos);

    if(GetKey(olc::Key::Q).bHeld || GetMouseWheel() > 0)
    {
      scale *= 1.1;
    }
    if(GetKey(olc::Key::A).bHeld || GetMouseWheel() < 0)
    {
      scale *= 0.9;
    }

    auto mouse_pos_after_zoom = screen_to_world(mouse_pos);

    camera_offset += (mouse_pos_before_zoom - mouse_pos_after_zoom);

    const auto screen_top_left = Vector2ui{0u, 0u};
    const auto screen_bottom_right = Vector2ui{screen_width, screen_height};

    auto fractal_top_left = screen_to_world(screen_top_left);

    // Handle User Input
    if(GetKey(olc::K1).bPressed)
    {
      method = 0;
    }
    if(GetKey(olc::K2).bPressed)
    {
      method = 1;
    }
    if(GetKey(olc::K3).bPressed)
    {
      method = 2;
    }
    if(GetKey(olc::K4).bPressed)
    {
      method = 3;
    }
    if(GetKey(olc::K5).bPressed)
    {
      method = 4;
    }
    if(GetKey(olc::K6).bPressed)
    {
      method = 5;
    }
    if(GetKey(olc::UP).bPressed)
    {
      iterations += 64;
    }
    if(GetKey(olc::DOWN).bPressed)
    {
      iterations -= 64;
    }
    if(GetKey(olc::LEFT).bPressed)
    {
      n_threads -= 1;
    }
    if(GetKey(olc::RIGHT).bPressed)
    {
      n_threads += 1;
    }
    iterations = std::max(iterations, 64u);
    n_threads = std::clamp(n_threads, 1u, max_threads);

    // Start timing
    auto tp1 = std::chrono::high_resolution_clock::now();

    switch(method)
    {
    case 0:
    {
      iterate_vanilla(screen_top_left, screen_bottom_right, fractal_top_left, 1.0 / scale.x,
                      1.0 / scale.y, iterations);
      break;
    }
    case 1:
    {
      deploy_threads(std::mem_fn(&Fractal_viewer::iterate_vanilla), screen_top_left,
                     screen_bottom_right, fractal_top_left, 1.0 / scale.x, 1.0 / scale.y,
                     iterations);
      break;
    }
    case 2:
    {
      deploy_thread_pool(std::mem_fn(&Fractal_viewer::iterate_vanilla), screen_top_left,
                         screen_bottom_right, fractal_top_left, 1.0 / scale.x, 1.0 / scale.y,
                         iterations);
      break;
    }
    case 3:
    {
      iterate_simd(screen_top_left, screen_bottom_right, fractal_top_left, 1.0 / scale.x,
                   1.0 / scale.y, iterations);
      break;
    }
    case 4:
    {
      deploy_threads(std::mem_fn(&Fractal_viewer::iterate_simd), screen_top_left,
                     screen_bottom_right, fractal_top_left, 1.0 / scale.x, 1.0 / scale.y,
                     iterations);
      break;
    }
    case 5:
    {
      deploy_thread_pool(std::mem_fn(&Fractal_viewer::iterate_simd), screen_top_left,
                         screen_bottom_right, fractal_top_left, 1.0 / scale.x, 1.0 / scale.y,
                         iterations);
      break;
    }
    }

    // Stop timing
    auto tp2 = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsedTime = tp2 - tp1;

    // Render
    for(uint y = 0; y < screen_height; y++)
    {
      for(uint x = 0; x < screen_width; x++)
      {
        auto colors = get_rgb(pixel_iterations[y * screen_width + x], iterations);
        Draw(x, y, olc::PixelF(std::get<0>(colors), std::get<1>(colors), std::get<2>(colors)));
      }
    }

    switch(method)
    {
    case 0:
    {
      DrawString(0, 0, "1) Vanilla", olc::WHITE, 3);
      break;
    }
    case 1:
    {
      DrawString(0, 0, "2) Vanilla with no. Threads: " + std::to_string(n_threads), olc::WHITE,
                 3);
      break;
    }
    case 2:
    {
      DrawString(0, 0, "3) Vanilla with Thread Pool, size: " + std::to_string(n_threads),
                 olc::WHITE, 3);
      break;
    }
    case 3:
    {
      DrawString(0, 0, "4) Intrinsics = AVX2", olc::WHITE, 3);
      break;
    }
    case 4:
    {
      DrawString(0, 0, "5) Intrinsics with no. Threads: " + std::to_string(n_threads),
                 olc::WHITE, 3);
      break;
    }
    case 5:
    {
      DrawString(0, 0, "6) Intrinsics with Thread Poo, size: " + std::to_string(n_threads),
                 olc::WHITE, 3);
      break;
    }
    }

    DrawString(0, 30,
               "Time taken for current frame: " + std::to_string(elapsedTime.count()) + "s",
               olc::WHITE, 3);
    DrawString(0, 60, "Current maximum iterations: " + std::to_string(iterations), olc::WHITE,
               3);
    return !(GetKey(olc::Key::ESCAPE).bPressed);
  }

  Vector2ui world_to_screen(const Vector2d& v)
  {
    return {(uint)((v.x - camera_offset.x) * scale.x),
            (uint)((v.y - camera_offset.y) * scale.y)};
  }

  Vector2d screen_to_world(const olc::vi2d& n)
  {
    return {(double)(n.x) / scale.x + camera_offset.x,
            (double)(n.y) / scale.y + camera_offset.y};
  }

  // static variables
  static constexpr uint max_threads = 8;
  static constexpr uint screen_width = 1920;
  static constexpr uint screen_height = 1080;
  static constexpr uint screen_size = screen_width * screen_height;
  static std::array<std::thread, 7> thread_pool;

  // member variables
  uint* pixel_iterations = nullptr;
  Vector2d camera_offset = {0.0, 0.0};
  Vector2d panning_pivot = {0.0, 0.0};
  Vector2d scale = {1280.0 / 2.0, 720.0};
  uint method = 0;
  uint n_threads = 8;
  uint iterations = 64;
};

std::array<std::thread, 7> Fractal_viewer::thread_pool;

int main()
{
  Fractal_viewer demo;
  if(demo.Construct(Fractal_viewer::screen_width, Fractal_viewer::screen_height, 1, 1, false,
                    false))
    demo.Start();
  return 0;
}
