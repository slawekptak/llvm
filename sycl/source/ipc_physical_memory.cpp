//==------- ipc_physical_memory.cpp -- SYCL inter-process for physical mem -==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <detail/adapter_impl.hpp>
#include <detail/context_impl.hpp>
#include <detail/physical_mem_impl.hpp>
#include <sycl/context.hpp>
#include <sycl/ext/oneapi/experimental/ipc_physical_memory.hpp>

#include <cstring>

namespace sycl {
inline namespace _V1 {

namespace detail {

__SYCL_EXPORT sycl::ext::oneapi::experimental::physical_mem
openIPCPhysicalMemHandle(const std::byte *HandleData, size_t HandleDataSize,
                         const sycl::context &Ctx, const sycl::device &Dev) {
  if (!Dev.has(aspect::ext_oneapi_ipc_physical_memory))
    throw sycl::exception(
        sycl::make_error_code(errc::feature_not_supported),
        "Device does not support aspect::ext_oneapi_ipc_physical_memory.");

  auto CtxImpl = sycl::detail::getSyclObjImpl(Ctx);
  sycl::detail::adapter_impl &Adapter = CtxImpl->getAdapter();

  // Extract size_t from the beginning of the buffer
  size_t NumBytes;
  std::memcpy(&NumBytes, HandleData, sizeof(size_t));

  // Actual UR handle starts after the size_t prefix
  const std::byte *ActualHandleData = HandleData + sizeof(size_t);
  size_t ActualHandleSize = HandleDataSize - sizeof(size_t);

  // TODO: UMF and UR currently requires the handle data to be non-const, so we
  //       need const-cast the data pointer. Once this has been changed, the
  //       const-cast can be removed.
  std::byte *NonConstActualHandleData =
      const_cast<std::byte *>(ActualHandleData);

  ur_physical_mem_handle_t PhysMemHandle = nullptr;
  ur_result_t UrRes = UR_RESULT_SUCCESS;
  // TODO IPC physical memory
  /*
      Adapter.call_nocheck<sycl::detail::UrApiKind::urIPCOpenPhysMemHandleExp>(
          CtxImpl->getHandleRef(), getSyclObjImpl(Dev)->getHandleRef(),
          NonConstActualHandleData, ActualHandleSize, &PhysMemHandle);
  */
  if (UrRes == UR_RESULT_ERROR_INVALID_VALUE)
    throw sycl::exception(
        sycl::make_error_code(errc::invalid),
        "HandleData data size does not correspond to the target platform's "
        "IPC physical memory handle size.");
  Adapter.checkUrResult(UrRes);
  if (PhysMemHandle == nullptr)
    throw sycl::exception(
        sycl::make_error_code(errc::runtime),
        "urIPCOpenPhysMemHandleExp returned success but did not produce a "
        "valid physical memory handle.");

  auto PhysMemImpl = std::make_shared<sycl::detail::physical_mem_impl>(
      *getSyclObjImpl(Dev), Ctx, NumBytes, PhysMemHandle);
  return sycl::detail::createSyclObjFromImpl<
      ext::oneapi::experimental::physical_mem>(PhysMemImpl);

  // return sycl::ext::oneapi::experimental::physical_mem{Dev, Ctx, 0};
}

} // namespace detail

namespace ext::oneapi::experimental::ipc::physical_memory {

__SYCL_EXPORT handle get(physical_mem &physmem) {
  if (!physmem.ipc_enabled())
    throw sycl::exception(
        sycl::make_error_code(errc::invalid),
        "physical_mem was not created with inter-process sharing enabled "
        "via the enable_ipc property.");

  auto PhysMemImpl = sycl::detail::getSyclObjImpl(physmem);
  auto CtxImpl = sycl::detail::getSyclObjImpl(physmem.get_context());
  sycl::detail::adapter_impl &Adapter = CtxImpl->getAdapter();

  // Get the UR handle from the adapter
  void *HandlePtr = nullptr;
  size_t HandleSize = 0;
  auto UrRes = UR_RESULT_SUCCESS;
  // TODO IPC physical memory
  /*
        Adapter.call_nocheck<sycl::detail::UrApiKind::urIPCGetPhysMemHandleExp>(
          CtxImpl->getHandleRef(), PhysMemImpl->getHandleRef(), &HandlePtr,
          &HandleSize);
      */
  Adapter.checkUrResult(UrRes);
  if (HandlePtr == nullptr)
    throw sycl::exception(
        sycl::make_error_code(errc::runtime),
        "urIPCGetPhysMemHandleExp returned success but did not produce a "
        "valid IPC handle.");

  // Get the size from the physical_mem object
  size_t NumBytes = physmem.size();

  // Allocate combined buffer: size_t + handle
  size_t TotalSize = sizeof(size_t) + HandleSize;
  std::byte *CombinedData = new std::byte[TotalSize];

  // Copy size first, then handle data
  std::memcpy(CombinedData, &NumBytes, sizeof(size_t));
  std::memcpy(CombinedData + sizeof(size_t), HandlePtr, HandleSize);

  return {CombinedData, TotalSize};
}

__SYCL_EXPORT void put(handle &ipc_handle, const sycl::context &ctx) {
  // Extract actual handle pointer (skip size_t prefix)
  std::byte *ActualHandleData = ipc_handle.MData + sizeof(size_t);

  auto CtxImpl = sycl::detail::getSyclObjImpl(ctx);
  sycl::detail::adapter_impl &Adapter = CtxImpl->getAdapter();

  ur_result_t UrRes = UR_RESULT_SUCCESS;
  // TODO IPC physical memory
  /*
  Adapter.call_nocheck<sycl::detail::UrApiKind::urIPCPutPhysMemHandleExp>(
      CtxImpl->getHandleRef(), ActualHandleData);
  */
  Adapter.checkUrResult(UrRes);
}

} // namespace ext::oneapi::experimental::ipc::physical_memory
} // namespace _V1
} // namespace sycl
