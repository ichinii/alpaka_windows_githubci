#include <alpaka/alpaka.hpp>

using Dim = alpaka::DimInt<1>;
using Idx = int32_t;
using Vec = alpaka::Vec<Dim, Idx>;

using AccCpu = alpaka::AccCpuSerial<Dim, Idx>;
using QueueCpu = alpaka::Queue<AccCpu, alpaka::NonBlocking>;

using AccGpu = alpaka::AccGpuCudaRt<Dim, Idx>;
using QueueGpu = alpaka::Queue<AccGpu, alpaka::NonBlocking>;

struct Kernel {
    template <typename TAcc>
    ALPAKA_FN_ACC
    void operator() (const TAcc& acc) const {
        using Idx = alpaka::Idx<TAcc>;
        const Idx gid = alpaka::getIdx<alpaka::Grid, alpaka::Threads>(acc)[0];
    }
};

int main([[maybe_unused]] int argc, [[maybe_unused]] char** argv) {
    const auto count = alpaka::getDevCount(alpaka::PlatformCpu());
    assert(0 < count);

    const auto N = static_cast<Idx>(1 << 10);
    const auto platform = alpaka::Platform<AccGpu>();
    const auto acc = alpaka::getDevByIdx(platform, 0);
    auto d_q = alpaka::Queue<AccGpu, alpaka::Blocking>(acc);

    alpaka::exec<AccGpu>(
        d_q,
        alpaka::getValidWorkDiv<AccGpu>(acc, N),
        Kernel{}
    );

    alpaka::wait(d_q);

    std::cout << "success" << std::endl;

    return 0;
}
