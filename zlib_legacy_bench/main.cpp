#define NOMINMAX

#include <algorithm>
#include <atomic>
#include <cstring>
#include <exception>
#include <iostream>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <thread>
#include <vector>
#include <zimg.h>

#include "aligned_malloc.h"
#include "argparse.h"
#include "timer.h"

namespace {;

const unsigned src_w = 1280;
const unsigned src_h = 720;
const unsigned src_w_uv = 640;
const unsigned src_h_uv = 360;
const unsigned dst_w = 1920;
const unsigned dst_h = 1080;
int floating_type = ZIMG_PIXEL_FLOAT;


int set_float_mode(const ArgparseOption *opt, void *, int, char **)
{
	if (!strcmp(opt->long_name, "half"))
		floating_type = ZIMG_PIXEL_HALF;
	else
		floating_type = ZIMG_PIXEL_FLOAT;

	return 0;
}


struct Context {
	std::unique_ptr<zimg_depth_context, decltype(&zimg_depth_delete)> to_f32{ nullptr, zimg_depth_delete };
	std::unique_ptr<zimg_resize_context, decltype(&zimg_resize_delete)> to_444{ nullptr, zimg_resize_delete };
	std::unique_ptr<zimg_colorspace_context, decltype(&zimg_colorspace_delete)> to_rgb{ nullptr, zimg_colorspace_delete };
	std::unique_ptr<zimg_resize_context, decltype(&zimg_resize_delete)> to_out{ nullptr, zimg_resize_delete };
	std::unique_ptr<zimg_depth_context, decltype(&zimg_depth_delete)> to_u8{ nullptr, zimg_depth_delete };
	size_t tmp_size;
};

struct PlaneBuffer {
	void *ptr;
	int stride;
	std::shared_ptr<void> handle;
};

void throw_zimg_error()
{
	char msg[1024];

	zimg_get_last_error(msg, sizeof(msg));
	throw std::runtime_error{ msg };
}

PlaneBuffer allocate_plane(unsigned w, unsigned h, int type)
{
	if (type == ZIMG_PIXEL_WORD || type == ZIMG_PIXEL_HALF)
		w *= 2;
	if (type == ZIMG_PIXEL_FLOAT)
		w *= 4;

	size_t rowsize = (w % 64) ? (w + 64 - w % 64) : w;
	std::shared_ptr<void> handle{ aligned_malloc(rowsize * h, 64), aligned_free };
	if (!handle)
		throw std::bad_alloc{};

	return{ handle.get(), (int)rowsize, handle };
}

void thread_target(Context *context, std::atomic_int *counter, std::exception_ptr *eptr, std::mutex *mutex)
{
	int err = 1;

	try {
		auto src_y = allocate_plane(src_w, src_h, ZIMG_PIXEL_WORD);
		auto src_u = allocate_plane(src_w_uv, src_h_uv, ZIMG_PIXEL_WORD);
		auto src_v = allocate_plane(src_w_uv, src_h_uv, ZIMG_PIXEL_WORD);

		auto srcfp_y = allocate_plane(src_w, src_h, floating_type);
		auto srcfp_u = allocate_plane(src_w_uv, src_h_uv, floating_type);
		auto srcfp_v = allocate_plane(src_w_uv, src_h_uv, floating_type);

		auto srcfp_444_u = allocate_plane(src_w, src_h, floating_type);
		auto srcfp_444_v = allocate_plane(src_w, src_h, floating_type);

		auto dstfp_y = allocate_plane(dst_w, dst_h, floating_type);
		auto dstfp_u = allocate_plane(dst_w, dst_h, floating_type);
		auto dstfp_v = allocate_plane(dst_w, dst_h, floating_type);

		auto dst_y = allocate_plane(dst_w, dst_h, ZIMG_PIXEL_BYTE);
		auto dst_u = allocate_plane(dst_w, dst_h, ZIMG_PIXEL_BYTE);
		auto dst_v = allocate_plane(dst_w, dst_h, ZIMG_PIXEL_BYTE);

		auto tmp = allocate_plane((unsigned)context->tmp_size, 1, ZIMG_PIXEL_BYTE);

		while (true) {
			if ((*counter)-- <= 0)
				break;

			if (zimg_depth_process(context->to_f32.get(), src_y.ptr, srcfp_y.ptr, tmp.ptr, src_w, src_h, src_y.stride, srcfp_y.stride,
			                       ZIMG_PIXEL_WORD, floating_type, 8, 32, 0, 0, 0))
				throw_zimg_error();

			if (zimg_depth_process(context->to_f32.get(), src_u.ptr, srcfp_u.ptr, tmp.ptr, src_w_uv, src_h_uv, src_u.stride, srcfp_u.stride,
			                       ZIMG_PIXEL_WORD, floating_type, 8, 32, 0, 0, 1))
				throw_zimg_error();

			if (zimg_depth_process(context->to_f32.get(), src_v.ptr, srcfp_v.ptr, tmp.ptr, src_w_uv, src_h_uv, src_v.stride, srcfp_v.stride,
			                       ZIMG_PIXEL_WORD, floating_type, 8, 32, 0, 0, 1))
				throw_zimg_error();

			if (zimg_resize_process(context->to_444.get(), srcfp_u.ptr, srcfp_444_u.ptr, tmp.ptr,
			                        src_w_uv, src_h_uv, src_w, src_h, srcfp_u.stride, srcfp_444_u.stride, floating_type))
				throw_zimg_error();

			if (zimg_resize_process(context->to_444.get(), srcfp_v.ptr, srcfp_444_v.ptr, tmp.ptr,
			                        src_w_uv, src_h_uv, src_w, src_h, srcfp_v.stride, srcfp_444_v.stride, floating_type))
				throw_zimg_error();

			void *planes_444[3] = { srcfp_y.ptr, srcfp_444_u.ptr, srcfp_444_v.ptr };
			int strides_444[3] = { srcfp_y.stride, srcfp_444_u.stride, srcfp_444_v.stride };

			if (zimg_colorspace_process(context->to_rgb.get(), planes_444, planes_444, tmp.ptr,
			                            src_w, src_h, strides_444, strides_444, floating_type))
				throw_zimg_error();

			if (zimg_resize_process(context->to_out.get(), srcfp_y.ptr, dstfp_y.ptr, tmp.ptr,
			                        src_w, src_h, dst_w, dst_h, srcfp_y.stride, dstfp_y.stride, floating_type))
				throw_zimg_error();

			if (zimg_resize_process(context->to_out.get(), srcfp_444_u.ptr, dstfp_u.ptr, tmp.ptr,
			                        src_w, src_h, dst_w, dst_h, srcfp_444_u.stride, dstfp_u.stride, floating_type))
				throw_zimg_error();

			if (zimg_resize_process(context->to_out.get(), srcfp_444_v.ptr, dstfp_v.ptr, tmp.ptr,
			                        src_w, src_h, dst_w, dst_h, srcfp_444_v.stride, dstfp_v.stride, floating_type))
				throw_zimg_error();

			if (zimg_depth_process(context->to_u8.get(), dstfp_y.ptr, dst_y.ptr, tmp.ptr, dst_w, dst_h, dstfp_y.stride, dst_y.stride,
			                       floating_type, ZIMG_PIXEL_BYTE, 32, 8, 0, 1, 0))
				throw_zimg_error();

			if (zimg_depth_process(context->to_u8.get(), dstfp_u.ptr, dst_u.ptr, tmp.ptr, dst_w, dst_h, dstfp_u.stride, dst_u.stride,
			                       floating_type, ZIMG_PIXEL_BYTE, 32, 8, 0, 1, 1))
				throw_zimg_error();

			if (zimg_depth_process(context->to_u8.get(), dstfp_v.ptr, dst_v.ptr, tmp.ptr, dst_w, dst_h, dstfp_v.stride, dst_v.stride,
			                       floating_type, ZIMG_PIXEL_BYTE, 32, 8, 0, 1, 1))
				throw_zimg_error();
		}
	} catch (...) {
		std::lock_guard<std::mutex>{ *mutex };
		*eptr = std::current_exception();
	}
}

void execute(unsigned times, unsigned threads)
{
	unsigned thread_min = threads ? threads : 1;
	unsigned thread_max = threads ? threads : std::thread::hardware_concurrency();

	Context context;

	context.to_f32.reset(zimg_depth_create(ZIMG_DITHER_ORDERED));
	if (!context.to_f32)
		throw_zimg_error();

	context.to_444.reset(zimg_resize_create(ZIMG_RESIZE_LANCZOS, src_w_uv, src_h_uv, src_w, src_h,
	                                        0.25, 0.0, src_w_uv, src_h_uv, 4.0, NAN));
	if (!context.to_444)
		throw_zimg_error();

	context.to_rgb.reset(zimg_colorspace_create(ZIMG_MATRIX_709, ZIMG_TRANSFER_709, ZIMG_PRIMARIES_709,
	                                            ZIMG_MATRIX_RGB, ZIMG_TRANSFER_709, ZIMG_PRIMARIES_709));
	if (!context.to_rgb)
		throw_zimg_error();

	context.to_out.reset(zimg_resize_create(ZIMG_RESIZE_LANCZOS, src_w, src_h, dst_w, dst_h,
	                                        0.0, 0.0, src_w, src_h, 4.0, NAN));
	if (!context.to_out)
		throw_zimg_error();

	context.to_u8.reset(zimg_depth_create(ZIMG_DITHER_ORDERED));
	if (!context.to_u8)
		throw_zimg_error();

	context.tmp_size = std::max(context.tmp_size, zimg_depth_tmp_size(context.to_f32.get(), src_w));
	context.tmp_size = std::max(context.tmp_size, zimg_depth_tmp_size(context.to_f32.get(), src_w_uv));
	context.tmp_size = std::max(context.tmp_size, zimg_resize_tmp_size(context.to_444.get(), floating_type));
	context.tmp_size = std::max(context.tmp_size, zimg_colorspace_tmp_size(context.to_rgb.get(), src_w));
	context.tmp_size = std::max(context.tmp_size, zimg_resize_tmp_size(context.to_out.get(), floating_type));
	context.tmp_size = std::max(context.tmp_size, zimg_depth_tmp_size(context.to_u8.get(), dst_w));

	std::cout << "1280x720/420/W => 1920x1080/RGB/B\n";

	for (unsigned n = thread_min; n <= thread_max; ++n) {
		std::vector<std::thread> thread_pool;
		std::atomic_int counter{ static_cast<int>(times * n) };
		std::exception_ptr eptr{};
		std::mutex mutex;
		Timer timer;

		timer.start();
		for (unsigned nn = 0; nn < n; ++nn) {
			thread_pool.emplace_back(thread_target, &context, &counter, &eptr, &mutex);
		}

		for (auto &th : thread_pool) {
			th.join();
		}
		timer.stop();

		if (eptr)
			std::rethrow_exception(eptr);

		std::cout << '\n';
		std::cout << "threads:    " << n << '\n';
		std::cout << "iterations: " << times * n << '\n';
		std::cout << "fps:        " << (times * n) / timer.elapsed() << '\n';
	}
}


struct Arguments {
	unsigned times;
	unsigned threads;
};

const ArgparseOption program_switches[] = {
	{ OPTION_USER,     nullptr, "half",    0,                            set_float_mode, "use half precision" },
	{ OPTION_USER,     nullptr, "float",   0,                            set_float_mode, "use single precision" },
	{ OPTION_UINTEGER, nullptr, "times",   offsetof(Arguments, times),   nullptr, "number of benchmark cycles per thread" },
	{ OPTION_UINTEGER, nullptr, "threads", offsetof(Arguments, threads), nullptr, "number of threads" },
};

const ArgparseCommandLine program_def = {
	program_switches,
	sizeof(program_switches) / sizeof(program_switches[0]),
	nullptr,
	0,
	"zlib_legacy_bench",
	"benchmark z.lib API v1",
	nullptr
};

} // namespace


int main(int argc, char **argv)
{
	Arguments args{};
	int ret;

	args.times = 100;

	if ((ret = argparse_parse(&program_def, &args, argc, argv)))
		return ret == ARGPARSE_HELP ? 0 : ret;

	zimg_set_cpu(ZIMG_CPU_AUTO);

	try {
		execute(args.times, args.threads);
	} catch (const std::runtime_error &e) {
		std::cerr << "runtime error: " << e.what() << '\n';
		return 1;
	}

	return 0;
}
