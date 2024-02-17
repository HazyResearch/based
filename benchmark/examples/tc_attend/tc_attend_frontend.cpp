#include <torch/extension.h>
#include <vector>

extern void qkv_fuse(torch::Tensor x, torch::Tensor q, torch::Tensor k, torch::Tensor v, 
                                    torch::Tensor q_out, torch::Tensor k_out, torch::Tensor v_out);
extern void  sliding_window(int j, torch::Tensor q, torch::Tensor k, torch::Tensor v, torch::Tensor v_out);
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.doc() = "Test handler for warp test"; // optional module docstring
    m.def("qkv_fuse", qkv_fuse);
    m.def("sliding_window", sliding_window);
}
