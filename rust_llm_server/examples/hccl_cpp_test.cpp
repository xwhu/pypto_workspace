#include <iostream>
#include <vector>
#include <unistd.h>
#include "acl/acl.h"
#include "hccl/hccl.h"
#include "hccl/hccl_types.h"

#define CHECK_ACL(expr) \
    do { \
        aclError ret = (expr); \
        if (ret != ACL_SUCCESS) { \
            std::cerr << "ACL Error at " << __FILE__ << ":" << __LINE__ << " code=" << ret << std::endl; \
            exit(1); \
        } \
    } while (0)

#define CHECK_HCCL(expr) \
    do { \
        HcclResult ret = (expr); \
        if (ret != HCCL_SUCCESS) { \
            std::cerr << "HCCL Error at " << __FILE__ << ":" << __LINE__ << " code=" << ret << std::endl; \
            exit(1); \
        } \
    } while (0)

int main(int argc, char** argv) {
    int rank = -1;
    int nranks = 2;
    for (int i = 1; i < argc; i++) {
        if (std::string(argv[i]) == "--rank" && i + 1 < argc) rank = std::stoi(argv[i+1]);
        if (std::string(argv[i]) == "--nranks" && i + 1 < argc) nranks = std::stoi(argv[i+1]);
    }

    if (rank == -1) {
        std::cerr << "Usage: --rank <rank> --nranks <nranks>" << std::endl;
        return 1;
    }

    const char* dev_env = getenv("ASCEND_DEVICE_ID");
    int dev_id = dev_env ? std::stoi(dev_env) : rank;

    std::cout << "[rank " << rank << "] Init ACL on device " << dev_id << std::endl;
    CHECK_ACL(aclInit(nullptr));
    CHECK_ACL(aclrtSetDevice(dev_id));

    aclrtStream stream;
    CHECK_ACL(aclrtCreateStream(&stream));

    // Shared root info file
    const char* root_file = "/tmp/hccl_cpp_smoke_root.bin";
    HcclRootInfo root_info;

    if (rank == 0) {
        remove(root_file);
        CHECK_HCCL(HcclGetRootInfo(&root_info));
        FILE* f = fopen(root_file, "wb");
        fwrite(&root_info, 1, sizeof(root_info), f);
        fclose(f);
        std::cout << "[rank 0] Root info written" << std::endl;
    } else {
        std::cout << "[rank " << rank << "] Waiting for root info..." << std::endl;
        while (access(root_file, F_OK) != 0) {
            usleep(100000); // 100ms
        }
        FILE* f = fopen(root_file, "rb");
        fread(&root_info, 1, sizeof(root_info), f);
        fclose(f);
        std::cout << "[rank " << rank << "] Root info read" << std::endl;
    }

    HcclComm comm;
    std::cout << "[rank " << rank << "] Calling HcclCommInitRootInfo..." << std::endl;
    CHECK_HCCL(HcclCommInitRootInfo(nranks, &root_info, rank, &comm));
    std::cout << "[rank " << rank << "] Comm initialized" << std::endl;

    if (rank == 0) remove(root_file);

    int count = 256;
    float* send_buf;
    float* recv_buf;
    CHECK_ACL(aclrtMalloc((void**)&send_buf, count * sizeof(float), ACL_MEM_MALLOC_NORMAL_ONLY));
    CHECK_ACL(aclrtMalloc((void**)&recv_buf, count * sizeof(float), ACL_MEM_MALLOC_NORMAL_ONLY));

    std::vector<float> host_data(count, rank + 1.0f);
    CHECK_ACL(aclrtMemcpy(send_buf, count * sizeof(float), host_data.data(), count * sizeof(float), ACL_MEMCPY_HOST_TO_DEVICE));

    std::cout << "[rank " << rank << "] Testing In-Place AllReduce" << std::endl;
    CHECK_HCCL(HcclAllReduce(send_buf, send_buf, count, HCCL_DATA_TYPE_FP32, HCCL_REDUCE_SUM, comm, stream));
    
    std::cout << "[rank " << rank << "] Syncing stream" << std::endl;
    CHECK_ACL(aclrtSynchronizeStream(stream));

    std::vector<float> result(count, 0.0f);
    CHECK_ACL(aclrtMemcpy(result.data(), count * sizeof(float), send_buf, count * sizeof(float), ACL_MEMCPY_DEVICE_TO_HOST));
    std::cout << "[rank " << rank << "] In-Place AllReduce result[0] = " << result[0] << std::endl;

    std::cout << "[rank " << rank << "] Testing Broadcast" << std::endl;
    if (rank == 0) {
        std::vector<float> bcast_data(count, 42.0f);
        CHECK_ACL(aclrtMemcpy(recv_buf, count * sizeof(float), bcast_data.data(), count * sizeof(float), ACL_MEMCPY_HOST_TO_DEVICE));
    }
    CHECK_HCCL(HcclBroadcast(recv_buf, count, HCCL_DATA_TYPE_FP32, 0, comm, stream));
    CHECK_ACL(aclrtSynchronizeStream(stream));

    CHECK_ACL(aclrtMemcpy(result.data(), count * sizeof(float), recv_buf, count * sizeof(float), ACL_MEMCPY_DEVICE_TO_HOST));
    std::cout << "[rank " << rank << "] Broadcast result[0] = " << result[0] << std::endl;

    HcclCommDestroy(comm);
    aclrtDestroyStream(stream);
    aclrtResetDevice(dev_id);
    aclFinalize();

    return 0;
}
