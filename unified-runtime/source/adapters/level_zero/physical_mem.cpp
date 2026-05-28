//===---------------- physical_mem.cpp - Level Zero Adapter ---------------===//
//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM
// Exceptions. See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "physical_mem.hpp"
#include "common.hpp"
#include "device.hpp"

#ifdef UR_ADAPTER_LEVEL_ZERO_V2
#include "v2/context.hpp"
#else
#include "context.hpp"
#endif

namespace ur::level_zero {

ur_result_t urPhysicalMemCreate(
    ur_context_handle_t hContext, ur_device_handle_t hDevice, size_t size,
    [[maybe_unused]] const ur_physical_mem_properties_t *pProperties,
    ur_physical_mem_handle_t *phPhysicalMem) {
  ZeStruct<ze_physical_mem_desc_t> PhysicalMemDesc;
  PhysicalMemDesc.flags = 0;
  PhysicalMemDesc.size = size;

  bool EnableIpc =
      pProperties && (pProperties->flags & UR_PHYSICAL_MEM_FLAG_ENABLE_IPC);

  ze_physical_mem_handle_t ZePhysicalMem;
  ZE2UR_CALL(zePhysicalMemCreate, (hContext->getZeHandle(), hDevice->ZeDevice,
                                   &PhysicalMemDesc, &ZePhysicalMem));
  try {
    *phPhysicalMem = new ur_physical_mem_handle_t_(ZePhysicalMem, hContext,
                                                   hDevice, size, EnableIpc);
  } catch (const std::bad_alloc &) {
    zePhysicalMemDestroy(hContext->getZeHandle(), ZePhysicalMem);
    return UR_RESULT_ERROR_OUT_OF_HOST_MEMORY;
  } catch (...) {
    zePhysicalMemDestroy(hContext->getZeHandle(), ZePhysicalMem);
    return UR_RESULT_ERROR_UNKNOWN;
  }
  return UR_RESULT_SUCCESS;
}

ur_result_t urPhysicalMemRetain(ur_physical_mem_handle_t hPhysicalMem) {
  hPhysicalMem->RefCount.retain();
  return UR_RESULT_SUCCESS;
}

ur_result_t urPhysicalMemRelease(ur_physical_mem_handle_t hPhysicalMem) {
  if (!hPhysicalMem->RefCount.release())
    return UR_RESULT_SUCCESS;

  if (hPhysicalMem->IpcVirtualAddress) {
    // IPC-opened handle: close the IPC virtual mapping instead of destroying
    // a physical mem handle (there is no ZePhysicalMem on the consumer side).
    if (checkL0LoaderTeardown()) {
      ZE2UR_CALL(zeMemCloseIpcHandle, (hPhysicalMem->Context->getZeHandle(),
                                       hPhysicalMem->IpcVirtualAddress));
    }
  } else if (hPhysicalMem->ZePhysicalMem) {
    if (checkL0LoaderTeardown()) {
      ZE2UR_CALL(zePhysicalMemDestroy, (hPhysicalMem->Context->getZeHandle(),
                                        hPhysicalMem->ZePhysicalMem));
    }
  }
  delete hPhysicalMem;

  return UR_RESULT_SUCCESS;
}

ur_result_t urPhysicalMemGetInfo(ur_physical_mem_handle_t hPhysicalMem,
                                 ur_physical_mem_info_t propName,
                                 size_t propSize, void *pPropValue,
                                 size_t *pPropSizeRet) {

  UrReturnHelper ReturnValue(propSize, pPropValue, pPropSizeRet);

  switch (propName) {
  case UR_PHYSICAL_MEM_INFO_CONTEXT:
    return ReturnValue(hPhysicalMem->Context);
  case UR_PHYSICAL_MEM_INFO_DEVICE:
    return ReturnValue(hPhysicalMem->Device);
  case UR_PHYSICAL_MEM_INFO_SIZE:
    return ReturnValue(hPhysicalMem->Size);
  case UR_PHYSICAL_MEM_INFO_PROPERTIES: {
    ur_physical_mem_flags_t Flags = static_cast<ur_physical_mem_flags_t>(0);
    if (hPhysicalMem->EnableIpc)
      Flags = UR_PHYSICAL_MEM_FLAG_ENABLE_IPC;
    ur_physical_mem_properties_t Props = {
        UR_STRUCTURE_TYPE_PHYSICAL_MEM_PROPERTIES, nullptr, Flags};
    return ReturnValue(Props);
  }
  case UR_PHYSICAL_MEM_INFO_REFERENCE_COUNT:
    return ReturnValue(hPhysicalMem->RefCount.getCount());
  default:
    return UR_RESULT_ERROR_UNSUPPORTED_ENUMERATION;
  }
  return UR_RESULT_SUCCESS;
}

ur_result_t urIPCGetPhysMemHandleExp(ur_context_handle_t hContext,
                                     ur_physical_mem_handle_t hPhysMem,
                                     void **ppIPCPhysMemHandleData,
                                     size_t *pIPCPhysMemHandleDataSizeRet) {
#ifdef __linux__
  if (!hContext)
    return UR_RESULT_ERROR_INVALID_NULL_HANDLE;
  if (!hPhysMem)
    return UR_RESULT_ERROR_INVALID_NULL_HANDLE;
  if (!ppIPCPhysMemHandleData)
    return UR_RESULT_ERROR_INVALID_NULL_POINTER;
  if (!pIPCPhysMemHandleDataSizeRet)
    return UR_RESULT_ERROR_INVALID_NULL_POINTER;
  if (!hPhysMem->EnableIpc)
    return UR_RESULT_ERROR_INVALID_ARGUMENT;

  // Cast the physical mem handle to const void* as required by
  // zeMemGetIpcHandleWithProperties.  The handle is passed without a virtual
  // mapping, which is the supported path for physical-mem IPC.
  ze_ipc_mem_handle_t IpcHandle = {};
  ZE2UR_CALL(zeMemGetIpcHandleWithProperties,
             (hContext->getZeHandle(),
              reinterpret_cast<const void *>(hPhysMem->ZePhysicalMem), nullptr,
              &IpcHandle));

  auto *HandleData = new (std::nothrow) ZeIPCPhysMemHandleData;
  if (!HandleData)
    return UR_RESULT_ERROR_OUT_OF_HOST_MEMORY;

  HandleData->IpcHandle = IpcHandle;
  HandleData->Size = hPhysMem->Size;

  *ppIPCPhysMemHandleData = HandleData;
  *pIPCPhysMemHandleDataSizeRet = sizeof(ZeIPCPhysMemHandleData);
  return UR_RESULT_SUCCESS;
#else
  (void)hContext;
  (void)hPhysMem;
  (void)ppIPCPhysMemHandleData;
  (void)pIPCPhysMemHandleDataSizeRet;
  return UR_RESULT_ERROR_UNSUPPORTED_FEATURE;
#endif // __linux__
}

ur_result_t urIPCPutPhysMemHandleExp(ur_context_handle_t hContext,
                                     const void *pIPCPhysMemHandleData) {
#ifdef __linux__
  if (!hContext)
    return UR_RESULT_ERROR_INVALID_NULL_HANDLE;
  if (!pIPCPhysMemHandleData)
    return UR_RESULT_ERROR_INVALID_NULL_POINTER;

  auto *HandleData =
      static_cast<const ZeIPCPhysMemHandleData *>(pIPCPhysMemHandleData);
  ZE2UR_CALL(zeMemPutIpcHandle,
             (hContext->getZeHandle(), HandleData->IpcHandle));
  delete HandleData;
  return UR_RESULT_SUCCESS;
#else
  (void)hContext;
  (void)pIPCPhysMemHandleData;
  return UR_RESULT_ERROR_UNSUPPORTED_FEATURE;
#endif // __linux__
}

ur_result_t urIPCOpenPhysMemHandleExp(ur_context_handle_t hContext,
                                      ur_device_handle_t hDevice,
                                      const void *pIPCPhysMemHandleData,
                                      size_t ipcPhysMemHandleDataSize,
                                      ur_physical_mem_handle_t *phPhysMem) {
#ifdef __linux__
  if (!hContext)
    return UR_RESULT_ERROR_INVALID_NULL_HANDLE;
  if (!hDevice)
    return UR_RESULT_ERROR_INVALID_NULL_HANDLE;
  if (!pIPCPhysMemHandleData)
    return UR_RESULT_ERROR_INVALID_NULL_POINTER;
  if (!phPhysMem)
    return UR_RESULT_ERROR_INVALID_NULL_POINTER;
  if (ipcPhysMemHandleDataSize != sizeof(ZeIPCPhysMemHandleData))
    return UR_RESULT_ERROR_INVALID_VALUE;

  auto *HandleData =
      static_cast<const ZeIPCPhysMemHandleData *>(pIPCPhysMemHandleData);

  // Open the IPC handle in this process.  zeMemOpenIpcHandle creates a virtual
  // mapping backed by the exporter's physical memory and returns a pointer to
  // it.  No separate zeVirtualMemMap call is needed; the memory is immediately
  // accessible at the returned address.
  void *VirtualAddress = nullptr;
  ZE2UR_CALL(zeMemOpenIpcHandle, (hContext->getZeHandle(), hDevice->ZeDevice,
                                  HandleData->IpcHandle, 0, &VirtualAddress));

  try {
    *phPhysMem = new ur_physical_mem_handle_t_(
        /*ZePhysicalMem=*/nullptr, hContext, hDevice, HandleData->Size,
        /*EnableIpc=*/true, VirtualAddress);
  } catch (const std::bad_alloc &) {
    zeMemCloseIpcHandle(hContext->getZeHandle(), VirtualAddress);
    return UR_RESULT_ERROR_OUT_OF_HOST_MEMORY;
  } catch (...) {
    zeMemCloseIpcHandle(hContext->getZeHandle(), VirtualAddress);
    return UR_RESULT_ERROR_UNKNOWN;
  }
  return UR_RESULT_SUCCESS;
#else
  (void)hContext;
  (void)hDevice;
  (void)pIPCPhysMemHandleData;
  (void)ipcPhysMemHandleDataSize;
  (void)phPhysMem;
  return UR_RESULT_ERROR_UNSUPPORTED_FEATURE;
#endif // __linux__
}

ur_result_t urIPCClosePhysMemHandleExp(ur_context_handle_t hContext,
                                       ur_physical_mem_handle_t hPhysMem) {
  if (!hContext)
    return UR_RESULT_ERROR_INVALID_NULL_HANDLE;
  if (!hPhysMem)
    return UR_RESULT_ERROR_INVALID_NULL_HANDLE;

  // Delegate to urPhysicalMemRelease so the refcount is respected.  For
  // IPC-opened handles (IpcVirtualAddress != nullptr) urPhysicalMemRelease
  // calls zeMemCloseIpcHandle; for regular handles it calls
  // zePhysicalMemDestroy.
  return ur::level_zero::urPhysicalMemRelease(hPhysMem);
}

} // namespace ur::level_zero
