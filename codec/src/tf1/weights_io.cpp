// Origin: fx2-cmix-transformer by Vladimir Ivanov, GPL-3.0, file cpp_infer/src/weights_io.cpp.
// Ported and modified for this entry in 2026; the changes are listed in
// sections 3 and 4 of README.md.
#include "weights_io.h"

#include <cstdarg>
#include <cstdio>
#include <cstdlib>
#include <cstring>

namespace fx2 {

namespace {

[[noreturn]] void die(unsigned code) {
  std::fprintf(stderr, "model error E%u\n", code);
  std::exit(1);
}

size_t dtype_size(uint8_t dtype) {
  switch (dtype) {
    case DT_I8:
      return 1;
    case DT_BF16:
      return 2;
    case DT_F32:
      return 4;
    case DT_I32:
      return 4;
    default:
      die(1);
  }
}

struct Reader {
  const uint8_t* p;
  size_t left;
  const char* path;

  void need(size_t n) {
    if (left < n) die(2);
  }
  void read(void* dst, size_t n) {
    need(n);
    std::memcpy(dst, p, n);
    p += n;
    left -= n;
  }
  uint32_t u32() {
    uint32_t v;
    read(&v, 4);
    return v;
  }
  uint8_t u8() {
    uint8_t v;
    read(&v, 1);
    return v;
  }
};

}  // namespace

WeightsFile WeightsFile::load(const char* path) {
  FILE* f = std::fopen(path, "rb");
  if (!f) die(3);
  std::fseek(f, 0, SEEK_END);
  long size = std::ftell(f);
  std::fseek(f, 0, SEEK_SET);
  if (size < 12) die(4);
  std::vector<uint8_t> buf(static_cast<size_t>(size));
  if (std::fread(buf.data(), 1, buf.size(), f) != buf.size())
    die(5);
  std::fclose(f);

  Reader r{buf.data(), buf.size(), path};
  char magic[8];
  r.read(magic, 8);
  if (std::memcmp(magic, "FX2TFW01", 8) != 0) die(6);

  uint32_t n_tensors = r.u32();
  WeightsFile wf;
  wf.tensors.reserve(n_tensors);
  for (uint32_t i = 0; i < n_tensors; i++) {
    uint32_t name_len = r.u32();
    if (name_len > 4096) die(7);
    std::string name(name_len, '\0');
    r.read(&name[0], name_len);

    WTensor t;
    t.dtype = r.u8();
    dtype_size(t.dtype);  // validates the code
    uint32_t ndim = r.u32();
    if (ndim > 8) die(8);
    t.shape.resize(ndim);
    size_t numel = 1;
    for (uint32_t d = 0; d < ndim; d++) {
      t.shape[d] = r.u32();
      numel *= t.shape[d];
    }
    t.numel = numel;
    size_t bytes = numel * dtype_size(t.dtype);
    t.data.resize(bytes);
    r.read(t.data.data(), bytes);

    if (wf.has(name))
      die(9);
    wf.tensors.emplace_back(name, std::move(t));
  }
  if (r.left != 0)
    die(10);
  return wf;
}

const WTensor& WeightsFile::get(std::string_view name) const {
  for (const auto& item : tensors)
    if (std::string_view(item.first) == name) return item.second;
  die(11);
}

const WTensor& WeightsFile::get(std::string_view name, uint8_t dtype,
                                std::initializer_list<uint32_t> shape) const {
  const WTensor& t = get(name);
  if (t.dtype != dtype)
    die(12);
  if (t.shape.size() != shape.size())
    die(13);
  size_t i = 0;
  for (uint32_t d : shape) {
    if (t.shape[i] != d)
      die(14);
    i++;
  }
  return t;
}

}  // namespace fx2

// Offline fatal-report map; no runtime diagnostic asset is required.
// E1: "\"unknown dtype code %u\", unsigned(dtype)"
// E2: "\"%s: truncated file (need %zu more bytes)\", path, n"
// E3: "\"cannot open %s\", path"
// E4: "\"%s: too small\", path"
// E5: "\"%s: short read\", path"
// E6: "\"%s: bad magic\", path"
// E7: "\"%s: absurd name length %u\", path, name_len"
// E8: "\"%s: %s: absurd ndim %u\", path, name.c_str(), ndim"
// E9: "\"%s: duplicate tensor %s\", path, name.c_str()"
// E10: "\"%s: %zu trailing bytes after last tensor\", path, r.left"
// E11: "\"missing tensor %s\", name.c_str()"
// E12: "\"%s: dtype %u, expected %u\", name.c_str(), unsigned(t.dtype),\n        unsigned(dtype)"
// E13: "\"%s: ndim %zu, expected %zu\", name.c_str(), t.shape.size(),\n        shape.size()"
// E14: "\"%s: shape[%zu] = %u, expected %u\", name.c_str(), i, t.shape[i], d"
