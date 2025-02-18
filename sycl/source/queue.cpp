//==-------------- queue.cpp -----------------------------------------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <detail/backend_impl.hpp>
#include <detail/event_impl.hpp>
#include <detail/queue_impl.hpp>
#include <sycl/detail/common.hpp>
#include <sycl/event.hpp>
#include <sycl/exception_list.hpp>
#include <sycl/handler.hpp>
#include <sycl/queue.hpp>
#include <detail/helpers.hpp>
#include <detail/cg.hpp>
#include <sycl/stream.hpp>

#include <algorithm>

namespace sycl {
inline namespace _V1 {

namespace detail {
SubmissionInfo::SubmissionInfo()
    : impl{std::make_shared<SubmissionInfoImpl>()} {}

optional<SubmitPostProcessF> &SubmissionInfo::PostProcessorFunc() {
  return impl->MPostProcessorFunc;
}

const optional<SubmitPostProcessF> &SubmissionInfo::PostProcessorFunc() const {
  return impl->MPostProcessorFunc;
}

std::shared_ptr<detail::queue_impl> &SubmissionInfo::SecondaryQueue() {
  return impl->MSecondaryQueue;
}

const std::shared_ptr<detail::queue_impl> &
SubmissionInfo::SecondaryQueue() const {
  return impl->MSecondaryQueue;
}

ext::oneapi::experimental::event_mode_enum &SubmissionInfo::EventMode() {
  return impl->MEventMode;
}

const ext::oneapi::experimental::event_mode_enum &
SubmissionInfo::EventMode() const {
  return impl->MEventMode;
}
} // namespace detail

queue::queue(const context &SyclContext, const device_selector &DeviceSelector,
             const async_handler &AsyncHandler, const property_list &PropList) {
  const std::vector<device> Devs = SyclContext.get_devices();

  auto Comp = [&DeviceSelector](const device &d1, const device &d2) {
    return DeviceSelector(d1) < DeviceSelector(d2);
  };

  const device &SyclDevice = *std::max_element(Devs.begin(), Devs.end(), Comp);

  impl = std::make_shared<detail::queue_impl>(
      detail::getSyclObjImpl(SyclDevice), detail::getSyclObjImpl(SyclContext),
      AsyncHandler, PropList);
}

queue::queue(const context &SyclContext, const device &SyclDevice,
             const async_handler &AsyncHandler, const property_list &PropList) {
  impl = std::make_shared<detail::queue_impl>(
      detail::getSyclObjImpl(SyclDevice), detail::getSyclObjImpl(SyclContext),
      AsyncHandler, PropList);
}

queue::queue(const device &SyclDevice, const async_handler &AsyncHandler,
             const property_list &PropList) {
  impl = std::make_shared<detail::queue_impl>(
      detail::getSyclObjImpl(SyclDevice), AsyncHandler, PropList);
}

queue::queue(const context &SyclContext, const device_selector &deviceSelector,
             const property_list &PropList)
    : queue(SyclContext, deviceSelector,
            detail::getSyclObjImpl(SyclContext)->get_async_handler(),
            PropList) {}

queue::queue(const context &SyclContext, const device &SyclDevice,
             const property_list &PropList)
    : queue(SyclContext, SyclDevice,
            detail::getSyclObjImpl(SyclContext)->get_async_handler(),
            PropList) {}

queue::queue(cl_command_queue clQueue, const context &SyclContext,
             const async_handler &AsyncHandler) {
  const property_list PropList{};
  impl = std::make_shared<detail::queue_impl>(
      // TODO(pi2ur): Don't cast straight from cl_command_queue
      reinterpret_cast<ur_queue_handle_t>(clQueue),
      detail::getSyclObjImpl(SyclContext), AsyncHandler, PropList);
}

cl_command_queue queue::get() const { return impl->get(); }

context queue::get_context() const { return impl->get_context(); }

device queue::get_device() const { return impl->get_device(); }

ext::oneapi::experimental::queue_state queue::ext_oneapi_get_state() const {
  return impl->getCommandGraph()
             ? ext::oneapi::experimental::queue_state::recording
             : ext::oneapi::experimental::queue_state::executing;
}

ext::oneapi::experimental::command_graph<
    ext::oneapi::experimental::graph_state::modifiable>
queue::ext_oneapi_get_graph() const {
  auto Graph = impl->getCommandGraph();
  if (!Graph)
    throw sycl::exception(
        make_error_code(errc::invalid),
        "ext_oneapi_get_graph() can only be called on recording queues.");

  return sycl::detail::createSyclObjFromImpl<
      ext::oneapi::experimental::command_graph<
          ext::oneapi::experimental::graph_state::modifiable>>(Graph);
}

void queue::throw_asynchronous() { impl->throw_asynchronous(); }

event queue::memset(void *Ptr, int Value, size_t Count,
                    const detail::code_location &CodeLoc) {
  detail::tls_code_loc_t TlsCodeLocCapture(CodeLoc);
  return impl->memset(impl, Ptr, Value, Count, {}, /*CallerNeedsEvent=*/true);
}

event queue::memset(void *Ptr, int Value, size_t Count, event DepEvent,
                    const detail::code_location &CodeLoc) {
  detail::tls_code_loc_t TlsCodeLocCapture(CodeLoc);
  return impl->memset(impl, Ptr, Value, Count, {DepEvent},
                      /*CallerNeedsEvent=*/true);
}

event queue::memset(void *Ptr, int Value, size_t Count,
                    const std::vector<event> &DepEvents,
                    const detail::code_location &CodeLoc) {
  detail::tls_code_loc_t TlsCodeLocCapture(CodeLoc);
  return impl->memset(impl, Ptr, Value, Count, DepEvents,
                      /*CallerNeedsEvent=*/true);
}

event queue::memcpy(void *Dest, const void *Src, size_t Count,
                    const detail::code_location &CodeLoc) {
  detail::tls_code_loc_t TlsCodeLocCapture(CodeLoc);
  return impl->memcpy(impl, Dest, Src, Count, {}, /*CallerNeedsEvent=*/true,
                      TlsCodeLocCapture.query());
}

event queue::memcpy(void *Dest, const void *Src, size_t Count, event DepEvent,
                    const detail::code_location &CodeLoc) {
  detail::tls_code_loc_t TlsCodeLocCapture(CodeLoc);
  return impl->memcpy(impl, Dest, Src, Count, {DepEvent},
                      /*CallerNeedsEvent=*/true, TlsCodeLocCapture.query());
}

event queue::memcpy(void *Dest, const void *Src, size_t Count,
                    const std::vector<event> &DepEvents,
                    const detail::code_location &CodeLoc) {
  detail::tls_code_loc_t TlsCodeLocCapture(CodeLoc);
  return impl->memcpy(impl, Dest, Src, Count, DepEvents,
                      /*CallerNeedsEvent=*/true, TlsCodeLocCapture.query());
}

event queue::mem_advise(const void *Ptr, size_t Length, int Advice,
                        const detail::code_location &CodeLoc) {
  detail::tls_code_loc_t TlsCodeLocCapture(CodeLoc);
  return impl->mem_advise(impl, Ptr, Length, ur_usm_advice_flags_t(Advice), {},
                          /*CallerNeedsEvent=*/true);
}

event queue::mem_advise(const void *Ptr, size_t Length, int Advice,
                        event DepEvent, const detail::code_location &CodeLoc) {
  detail::tls_code_loc_t TlsCodeLocCapture(CodeLoc);
  return impl->mem_advise(impl, Ptr, Length, ur_usm_advice_flags_t(Advice),
                          {DepEvent},
                          /*CallerNeedsEvent=*/true);
}

event queue::mem_advise(const void *Ptr, size_t Length, int Advice,
                        const std::vector<event> &DepEvents,
                        const detail::code_location &CodeLoc) {
  detail::tls_code_loc_t TlsCodeLocCapture(CodeLoc);
  return impl->mem_advise(impl, Ptr, Length, ur_usm_advice_flags_t(Advice),
                          DepEvents,
                          /*CallerNeedsEvent=*/true);
}

#ifndef __INTEL_PREVIEW_BREAKING_CHANGES
/// TODO: Unused. Remove these when ABI-break window is open.
event queue::submit_impl(std::function<void(handler &)> CGH,
                         const detail::code_location &CodeLoc) {
  return submit_with_event_impl(std::move(CGH), {}, CodeLoc, true);
}
event queue::submit_impl(std::function<void(handler &)> CGH,
                         const detail::code_location &CodeLoc,
                         bool IsTopCodeLoc) {
  return submit_with_event_impl(std::move(CGH), {}, CodeLoc, IsTopCodeLoc);
}

event queue::submit_impl(std::function<void(handler &)> CGH, queue SecondQueue,
                         const detail::code_location &CodeLoc) {
  return impl->submit(CGH, impl, SecondQueue.impl, CodeLoc, true);
}
event queue::submit_impl(std::function<void(handler &)> CGH, queue SecondQueue,
                         const detail::code_location &CodeLoc,
                         bool IsTopCodeLoc) {
  return impl->submit(CGH, impl, SecondQueue.impl, CodeLoc, IsTopCodeLoc);
}

void queue::submit_without_event_impl(std::function<void(handler &)> CGH,
                                      const detail::code_location &CodeLoc) {
  submit_without_event_impl(std::move(CGH), {}, CodeLoc, true);
}
void queue::submit_without_event_impl(std::function<void(handler &)> CGH,
                                      const detail::code_location &CodeLoc,
                                      bool IsTopCodeLoc) {
  submit_without_event_impl(std::move(CGH), {}, CodeLoc, IsTopCodeLoc);
}

event queue::submit_impl_and_postprocess(
    std::function<void(handler &)> CGH, const detail::code_location &CodeLoc,
    const detail::SubmitPostProcessF &PostProcess) {
  detail::SubmissionInfo SI{};
  SI.PostProcessorFunc() = std::move(PostProcess);
  return submit_with_event_impl(std::move(CGH), SI, CodeLoc, true);
}
event queue::submit_impl_and_postprocess(
    std::function<void(handler &)> CGH, const detail::code_location &CodeLoc,
    const detail::SubmitPostProcessF &PostProcess, bool IsTopCodeLoc) {
  detail::SubmissionInfo SI{};
  SI.PostProcessorFunc() = std::move(PostProcess);
  return submit_with_event_impl(std::move(CGH), SI, CodeLoc, IsTopCodeLoc);
}

event queue::submit_impl_and_postprocess(
    std::function<void(handler &)> CGH, queue SecondQueue,
    const detail::code_location &CodeLoc,
    const detail::SubmitPostProcessF &PostProcess) {
  return impl->submit(CGH, impl, SecondQueue.impl, CodeLoc, true, &PostProcess);
}
event queue::submit_impl_and_postprocess(
    std::function<void(handler &)> CGH, queue SecondQueue,
    const detail::code_location &CodeLoc,
    const detail::SubmitPostProcessF &PostProcess, bool IsTopCodeLoc) {
  return impl->submit(CGH, impl, SecondQueue.impl, CodeLoc, IsTopCodeLoc,
                      &PostProcess);
}

event queue::submit_with_event_impl(std::function<void(handler &)> CGH,
                                    const detail::SubmissionInfo &SubmitInfo,
                                    const detail::code_location &CodeLoc,
                                    bool IsTopCodeLoc) {
  return impl->submit_with_event(CGH, impl, SubmitInfo, CodeLoc, IsTopCodeLoc);
}

void queue::submit_without_event_impl(std::function<void(handler &)> CGH,
                                      const detail::SubmissionInfo &SubmitInfo,
                                      const detail::code_location &CodeLoc,
                                      bool IsTopCodeLoc) {
  impl->submit_without_event(CGH, impl, SubmitInfo, CodeLoc, IsTopCodeLoc);
}
#endif // __INTEL_PREVIEW_BREAKING_CHANGES

event queue::submit_with_event_impl(const detail::type_erased_cgfo_ty &CGH,
                                    const detail::SubmissionInfo &SubmitInfo,
                                    const detail::code_location &CodeLoc,
                                    bool IsTopCodeLoc) {
  return impl->submit_with_event(CGH, impl, SubmitInfo, CodeLoc, IsTopCodeLoc);
}

void queue::submit_without_event_impl(const detail::type_erased_cgfo_ty &CGH,
                                      const detail::SubmissionInfo &SubmitInfo,
                                      const detail::code_location &CodeLoc,
                                      bool IsTopCodeLoc) {
  impl->submit_without_event(CGH, impl, SubmitInfo, CodeLoc, IsTopCodeLoc);
}

void queue::wait_proxy(const detail::code_location &CodeLoc) {
  impl->wait(CodeLoc);
}

void queue::wait_and_throw_proxy(const detail::code_location &CodeLoc) {
  impl->wait_and_throw(CodeLoc);
}

static event
getBarrierEventForInorderQueueHelper(const detail::QueueImplPtr QueueImpl) {
  // This function should not be called when a queue is recording to a graph,
  // as a graph can record from multiple queues and we cannot guarantee the
  // last node added by an in-order queue will be the last node added to the
  // graph.
  assert(!QueueImpl->getCommandGraph() &&
         "Should not be called in on graph recording.");

  sycl::detail::optional<event> LastEvent = QueueImpl->getLastEvent();
  if (LastEvent)
    return *LastEvent;

  // If there was no last event, we create an empty one.
  return detail::createSyclObjFromImpl<event>(
      std::make_shared<detail::event_impl>(std::nullopt));
}

/// Prevents any commands submitted afterward to this queue from executing
/// until all commands previously submitted to this queue have entered the
/// complete state.
///
/// \param CodeLoc is the code location of the submit call (default argument)
/// \return a SYCL event object, which corresponds to the queue the command
/// group is being enqueued on.
event queue::ext_oneapi_submit_barrier(const detail::code_location &CodeLoc) {
  if (is_in_order() && !impl->getCommandGraph() && !impl->MDiscardEvents &&
      !impl->MIsProfilingEnabled) {
    event InOrderLastEvent = getBarrierEventForInorderQueueHelper(impl);
    // If the last event was discarded, fall back to enqueuing a barrier.
    if (!detail::getSyclObjImpl(InOrderLastEvent)->isDiscarded())
      return InOrderLastEvent;
  }

  return submit([=](handler &CGH) { CGH.ext_oneapi_barrier(); }, CodeLoc);
}

/// Prevents any commands submitted afterward to this queue from executing
/// until all events in WaitList have entered the complete state. If WaitList
/// is empty, then ext_oneapi_submit_barrier has no effect.
///
/// \param WaitList is a vector of valid SYCL events that need to complete
/// before barrier command can be executed.
/// \param CodeLoc is the code location of the submit call (default argument)
/// \return a SYCL event object, which corresponds to the queue the command
/// group is being enqueued on.
event queue::ext_oneapi_submit_barrier(const std::vector<event> &WaitList,
                                       const detail::code_location &CodeLoc) {
  bool AllEventsEmptyOrNop = std::all_of(
      begin(WaitList), end(WaitList), [&](const event &Event) -> bool {
        auto EventImpl = detail::getSyclObjImpl(Event);
        return (EventImpl->isDefaultConstructed() || EventImpl->isNOP()) &&
               !EventImpl->getCommandGraph();
      });
  if (is_in_order() && !impl->getCommandGraph() && !impl->MDiscardEvents &&
      !impl->MIsProfilingEnabled && AllEventsEmptyOrNop) {
    event InOrderLastEvent = getBarrierEventForInorderQueueHelper(impl);
    // If the last event was discarded, fall back to enqueuing a barrier.
    if (!detail::getSyclObjImpl(InOrderLastEvent)->isDiscarded())
      return InOrderLastEvent;
  }

  return submit([=](handler &CGH) { CGH.ext_oneapi_barrier(WaitList); },
                CodeLoc);
}

template <typename Param>
typename detail::is_queue_info_desc<Param>::return_type
queue::get_info() const {
  return impl->get_info<Param>();
}

#define __SYCL_PARAM_TRAITS_SPEC(DescType, Desc, ReturnT, Picode)              \
  template __SYCL_EXPORT ReturnT queue::get_info<info::queue::Desc>() const;

#include <sycl/info/queue_traits.def>

#undef __SYCL_PARAM_TRAITS_SPEC

template <typename Param>
typename detail::is_backend_info_desc<Param>::return_type
queue::get_backend_info() const {
  return impl->get_backend_info<Param>();
}

#define __SYCL_PARAM_TRAITS_SPEC(DescType, Desc, ReturnT, Picode)              \
  template __SYCL_EXPORT ReturnT                                               \
  queue::get_backend_info<info::DescType::Desc>() const;

#include <sycl/info/sycl_backend_traits.def>

#undef __SYCL_PARAM_TRAITS_SPEC

bool queue::is_in_order() const {
  return has_property<property::queue::in_order>();
}

backend queue::get_backend() const noexcept { return getImplBackend(impl); }

bool queue::ext_oneapi_empty() const { return impl->ext_oneapi_empty(); }

void queue::ext_oneapi_prod() { impl->flush(); }

ur_native_handle_t queue::getNative(int32_t &NativeHandleDesc) const {
  return impl->getNative(NativeHandleDesc);
}

event queue::memcpyToDeviceGlobal(void *DeviceGlobalPtr, const void *Src,
                                  bool IsDeviceImageScope, size_t NumBytes,
                                  size_t Offset,
                                  const std::vector<event> &DepEvents) {
  return impl->memcpyToDeviceGlobal(impl, DeviceGlobalPtr, Src,
                                    IsDeviceImageScope, NumBytes, Offset,
                                    DepEvents, /*CallerNeedsEvent=*/true);
}

event queue::memcpyFromDeviceGlobal(void *Dest, const void *DeviceGlobalPtr,
                                    bool IsDeviceImageScope, size_t NumBytes,
                                    size_t Offset,
                                    const std::vector<event> &DepEvents) {
  return impl->memcpyFromDeviceGlobal(impl, Dest, DeviceGlobalPtr,
                                      IsDeviceImageScope, NumBytes, Offset,
                                      DepEvents, /*CallerNeedsEvent=*/true);
}

bool queue::device_has(aspect Aspect) const {
  // avoid creating sycl object from impl
  return impl->getDeviceImplPtr()->has(Aspect);
}

// TODO(#15184) Remove this function in the next ABI-breaking window.
bool queue::ext_codeplay_supports_fusion() const { return false; }

sycl::detail::optional<event> queue::ext_oneapi_get_last_event_impl() const {
  if (!is_in_order())
    throw sycl::exception(
        make_error_code(errc::invalid),
        "ext_oneapi_get_last_event() can only be called on in-order queues.");

  sycl::detail::optional<event> LastEvent = impl->getLastEvent();

  // If there was no last event, the queue is yet to have any work submitted and
  // we return a std::nullopt.
  if (!LastEvent)
    return std::nullopt;

  // If the last event was discarded or a NOP, we insert a marker to represent
  // an event at end.
  auto LastEventImpl = detail::getSyclObjImpl(*LastEvent);
  if (LastEventImpl->isDiscarded() || LastEventImpl->isNOP())
    LastEvent =
        detail::createSyclObjFromImpl<event>(impl->insertMarkerEvent(impl));
  return LastEvent;
}

void queue::ext_oneapi_set_external_event(const event &external_event) {
  if (!is_in_order())
    throw sycl::exception(make_error_code(errc::invalid),
                          "ext_oneapi_set_external_event() can only be called "
                          "on in-order queues.");
  return impl->setExternalEvent(external_event);
}

const property_list &queue::getPropList() const { return impl->getPropList(); }

// from handler

template <typename KernelName, typename KernelType, typename PropertiesT,
          bool HasKernelHandlerArg, typename FuncTy>
void queue::unpack(_KERNELFUNCPARAM(KernelFunc), FuncTy Lambda) {
#ifdef __SYCL_DEVICE_ONLY__
  detail::CheckDeviceCopyable<KernelType>();
#endif // __SYCL_DEVICE_ONLY__
  using MergedPropertiesT =
      typename detail::GetMergedKernelProperties<KernelType,
                                                  PropertiesT>::type;
  using Unpacker = KernelPropertiesUnpacker<MergedPropertiesT>;
#ifndef __SYCL_DEVICE_ONLY__
  // If there are properties provided by get method then process them.
  /*
  if constexpr (ext::oneapi::experimental::detail::
                    HasKernelPropertiesGetMethod<
                        _KERNELFUNCPARAMTYPE>::value) {
    processProperties<detail::isKernelESIMD<KernelName>()>(
        KernelFunc.get(ext::oneapi::experimental::properties_tag{}));
  }
  */
#endif
  if constexpr (HasKernelHandlerArg) {
    kernel_handler KH;
    Lambda(Unpacker{}, this, KernelFunc, KH);
  } else {
    Lambda(Unpacker{}, this, KernelFunc);
  }
}

template <
    typename KernelName, typename KernelType,
    typename PropertiesT = ext::oneapi::experimental::empty_properties_t>
void queue::kernel_single_task_wrapper(_KERNELFUNCPARAM(KernelFunc)) {
  unpack<KernelName, KernelType, PropertiesT,
          detail::KernelLambdaHasKernelHandlerArgT<KernelType>::value>(
      KernelFunc, [&](auto Unpacker, auto... args) {
        Unpacker.template kernel_single_task_unpack<KernelName, KernelType>(
            args...);
      });
}

sycl::detail::CGType queue::getType() const { return impl->MCGType; }

void queue::throwIfActionIsCreated() {
  if (detail::CGType::None != getType())
    throw sycl::exception(make_error_code(errc::runtime),
                          "Attempt to set multiple actions for the "
                          "command group. Command group must consist of "
                          "a single kernel or explicit memory operation.");
}

#ifndef __SYCL_DEVICE_ONLY__
constexpr static int AccessTargetMask = 0x7ff;

template <typename KernelName, typename KernelType>
void queue::throwOnKernelParameterMisuse() const {
  using NameT =
      typename detail::get_kernel_name_t<KernelName, KernelType>::name;
  for (unsigned I = 0; I < detail::getKernelNumParams<NameT>(); ++I) {
    const detail::kernel_param_desc_t ParamDesc =
        detail::getKernelParamDesc<NameT>(I);
    const detail::kernel_param_kind_t &Kind = ParamDesc.kind;
    const access::target AccTarget =
        static_cast<access::target>(ParamDesc.info & AccessTargetMask);
    if ((Kind == detail::kernel_param_kind_t::kind_accessor) &&
        (AccTarget == target::local))
      throw sycl::exception(
          make_error_code(errc::kernel_argument),
          "A local accessor must not be used in a SYCL kernel function "
          "that is invoked via single_task or via the simple form of "
          "parallel_for that takes a range parameter.");
    if (Kind == detail::kernel_param_kind_t::kind_work_group_memory)
      throw sycl::exception(
          make_error_code(errc::kernel_argument),
          "A work group memory object must not be used in a SYCL kernel "
          "function that is invoked via single_task or via the simple form "
          "of parallel_for that takes a range parameter.");
  }
}
#endif

std::shared_ptr<detail::kernel_bundle_impl>
queue::getOrInsertHandlerKernelBundle(bool Insert) const {
  /*
  if (!impl->MKernelBundle && Insert) {
    auto Ctx =
        impl->MGraph ? impl->MGraph->getContext() : MQueue->get_context();
    auto Dev = impl->MGraph ? impl->MGraph->getDevice() : MQueue->get_device();
    impl->MKernelBundle = detail::getSyclObjImpl(
        get_kernel_bundle<bundle_state::input>(Ctx, {Dev}, {}));
  }
  */
  return impl->MKernelBundle;
}

void queue::verifyUsedKernelBundleInternal(detail::string_view KernelName) {
  auto UsedKernelBundleImplPtr =
      getOrInsertHandlerKernelBundle(/*Insert=*/false);
  if (!UsedKernelBundleImplPtr)
    return;

  // Implicit kernel bundles are populated late so we ignore them
  /*
  if (!impl->isStateExplicitKernelBundle())
    return;

  kernel_id KernelID = detail::get_kernel_id_impl(KernelName);
  device Dev = impl->MGraph ? impl->MGraph->getDevice()
                            : detail::getDeviceFromHandler(*this);
  if (!UsedKernelBundleImplPtr->has_kernel(KernelID, Dev))
    throw sycl::exception(
        make_error_code(errc::kernel_not_supported),
        "The kernel bundle in use does not contain the kernel");
  */
}

void queue::setNDRangeDescriptorPadded(sycl::range<3> N,
                                        bool SetNumWorkGroups, int Dims) {
  impl->MNDRDesc = detail::NDRDescT{N, SetNumWorkGroups, Dims};
}

void queue::clearArgs() { impl->MArgs.clear(); }

void queue::addArg(detail::kernel_param_kind_t ArgKind, void *Req,
                  int AccessTarget, int ArgIndex) {
  impl->MArgs.emplace_back(ArgKind, Req, AccessTarget, ArgIndex);
}

static void addArgsForGlobalAccessor(detail::Requirement *AccImpl, size_t Index,
                                    size_t &IndexShift, int Size,
                                    bool IsKernelCreatedFromSource,
                                    size_t GlobalSize,
                                    std::vector<detail::ArgDesc> &Args,
                                    bool isESIMD) {
  using detail::kernel_param_kind_t;
  if (AccImpl->PerWI)
    AccImpl->resize(GlobalSize);

  Args.emplace_back(kernel_param_kind_t::kind_accessor, AccImpl, Size,
                    Index + IndexShift);

  // TODO ESIMD currently does not suport offset, memory and access ranges -
  // accessor::init for ESIMD-mode accessor has a single field, translated
  // to a single kernel argument set above.
  if (!isESIMD && !IsKernelCreatedFromSource) {
    // Dimensionality of the buffer is 1 when dimensionality of the
    // accessor is 0.
    const size_t SizeAccField =
        sizeof(size_t) * (AccImpl->MDims == 0 ? 1 : AccImpl->MDims);
    ++IndexShift;
    Args.emplace_back(kernel_param_kind_t::kind_std_layout,
                      &AccImpl->MAccessRange[0], SizeAccField,
                      Index + IndexShift);
    ++IndexShift;
    Args.emplace_back(kernel_param_kind_t::kind_std_layout,
                      &AccImpl->MMemoryRange[0], SizeAccField,
                      Index + IndexShift);
    ++IndexShift;
    Args.emplace_back(kernel_param_kind_t::kind_std_layout,
                      &AccImpl->MOffset[0], SizeAccField, Index + IndexShift);
  }
}

void queue::processArg(void *Ptr, const detail::kernel_param_kind_t &Kind,
                        const int Size, const size_t Index, size_t &IndexShift,
                        bool IsKernelCreatedFromSource, bool IsESIMD) {
  using detail::kernel_param_kind_t;

  switch (Kind) {
  case kernel_param_kind_t::kind_std_layout:
  case kernel_param_kind_t::kind_pointer: {
    addArg(Kind, Ptr, Size, Index + IndexShift);
    break;
  }
  case kernel_param_kind_t::kind_stream: {
    // Stream contains several accessors inside.
    stream *S = static_cast<stream *>(Ptr);

    detail::AccessorBaseHost *GBufBase =
        static_cast<detail::AccessorBaseHost *>(&S->GlobalBuf);
    detail::AccessorImplPtr GBufImpl = detail::getSyclObjImpl(*GBufBase);
    detail::Requirement *GBufReq = GBufImpl.get();
    addArgsForGlobalAccessor(
        GBufReq, Index, IndexShift, Size, IsKernelCreatedFromSource,
        impl->MNDRDesc.GlobalSize.size(), impl->MArgs, IsESIMD);
    ++IndexShift;
    detail::AccessorBaseHost *GOffsetBase =
        static_cast<detail::AccessorBaseHost *>(&S->GlobalOffset);
    detail::AccessorImplPtr GOfssetImpl = detail::getSyclObjImpl(*GOffsetBase);
    detail::Requirement *GOffsetReq = GOfssetImpl.get();
    addArgsForGlobalAccessor(
        GOffsetReq, Index, IndexShift, Size, IsKernelCreatedFromSource,
        impl->MNDRDesc.GlobalSize.size(), impl->MArgs, IsESIMD);
    ++IndexShift;
    detail::AccessorBaseHost *GFlushBase =
        static_cast<detail::AccessorBaseHost *>(&S->GlobalFlushBuf);
    detail::AccessorImplPtr GFlushImpl = detail::getSyclObjImpl(*GFlushBase);
    detail::Requirement *GFlushReq = GFlushImpl.get();

    size_t GlobalSize = impl->MNDRDesc.GlobalSize.size();
    // If work group size wasn't set explicitly then it must be recieved
    // from kernel attribute or set to default values.
    // For now we can't get this attribute here.
    // So we just suppose that WG size is always default for stream.
    // TODO adjust MNDRDesc when device image contains kernel's attribute
    if (GlobalSize == 0) {
      // Suppose that work group size is 1 for every dimension
      GlobalSize = impl->MNDRDesc.NumWorkGroups.size();
    }
    addArgsForGlobalAccessor(GFlushReq, Index, IndexShift, Size,
                            IsKernelCreatedFromSource, GlobalSize, impl->MArgs,
                            IsESIMD);
    ++IndexShift;
    addArg(kernel_param_kind_t::kind_std_layout, &S->FlushBufferSize,
          sizeof(S->FlushBufferSize), Index + IndexShift);

    break;
  }
  case kernel_param_kind_t::kind_accessor: {
    // For args kind of accessor Size is information about accessor.
    // The first 11 bits of Size encodes the accessor target.
    const access::target AccTarget =
        static_cast<access::target>(Size & AccessTargetMask);
    switch (AccTarget) {
    case access::target::device:
    case access::target::constant_buffer: {
      detail::Requirement *AccImpl = static_cast<detail::Requirement *>(Ptr);
      addArgsForGlobalAccessor(
          AccImpl, Index, IndexShift, Size, IsKernelCreatedFromSource,
          impl->MNDRDesc.GlobalSize.size(), impl->MArgs, IsESIMD);
      break;
    }
    case access::target::local: {
      detail::LocalAccessorImplHost *LAcc =
          static_cast<detail::LocalAccessorImplHost *>(Ptr);

      range<3> &Size = LAcc->MSize;
      const int Dims = LAcc->MDims;
      int SizeInBytes = LAcc->MElemSize;
      for (int I = 0; I < Dims; ++I)
        SizeInBytes *= Size[I];
      // Some backends do not accept zero-sized local memory arguments, so we
      // make it a minimum allocation of 1 byte.
      SizeInBytes = std::max(SizeInBytes, 1);
      impl->MArgs.emplace_back(kernel_param_kind_t::kind_std_layout, nullptr,
                              SizeInBytes, Index + IndexShift);
      // TODO ESIMD currently does not suport MSize field passing yet
      // accessor::init for ESIMD-mode accessor has a single field, translated
      // to a single kernel argument set above.
      if (!IsESIMD && !IsKernelCreatedFromSource) {
        ++IndexShift;
        const size_t SizeAccField = (Dims == 0 ? 1 : Dims) * sizeof(Size[0]);
        addArg(kernel_param_kind_t::kind_std_layout, &Size, SizeAccField,
              Index + IndexShift);
        ++IndexShift;
        addArg(kernel_param_kind_t::kind_std_layout, &Size, SizeAccField,
              Index + IndexShift);
        ++IndexShift;
        addArg(kernel_param_kind_t::kind_std_layout, &Size, SizeAccField,
              Index + IndexShift);
      }
      break;
    }
    case access::target::image:
    case access::target::image_array: {
      detail::Requirement *AccImpl = static_cast<detail::Requirement *>(Ptr);
      addArg(Kind, AccImpl, Size, Index + IndexShift);
      if (!IsKernelCreatedFromSource) {
        // TODO Handle additional kernel arguments for image class
        // if the compiler front-end adds them.
      }
      break;
    }
    case access::target::host_image:
    case access::target::host_task:
    case access::target::host_buffer: {
      throw sycl::exception(make_error_code(errc::invalid),
                            "Unsupported accessor target case.");
      break;
    }
    }
    break;
  }
  case kernel_param_kind_t::kind_work_group_memory: {
    addArg(kernel_param_kind_t::kind_std_layout, nullptr,
          static_cast<detail::work_group_memory_impl *>(Ptr)->buffer_size,
          Index + IndexShift);
    break;
  }
  case kernel_param_kind_t::kind_sampler: {
    addArg(kernel_param_kind_t::kind_sampler, Ptr, sizeof(sampler),
          Index + IndexShift);
    break;
  }
  case kernel_param_kind_t::kind_specialization_constants_buffer: {
    addArg(kernel_param_kind_t::kind_specialization_constants_buffer, Ptr, Size,
          Index + IndexShift);
    break;
  }
  case kernel_param_kind_t::kind_invalid:
    throw exception(make_error_code(errc::invalid),
                    "Invalid kernel param kind");
    break;
  }
}

void queue::setArgsToAssociatedAccessors() {
  impl->MArgs = impl->MAssociatedAccesors;
}

inline constexpr size_t MaxNumAdditionalArgs = 13;

void queue::extractArgsAndReqsFromLambda(
    char *LambdaPtr, const std::vector<detail::kernel_param_desc_t> &ParamDescs,
    bool IsESIMD) {
  const bool IsKernelCreatedFromSource = false;
  size_t IndexShift = 0;
  impl->MArgs.reserve(MaxNumAdditionalArgs * ParamDescs.size());

  for (size_t I = 0; I < ParamDescs.size(); ++I) {
    void *Ptr = LambdaPtr + ParamDescs[I].offset;
    const detail::kernel_param_kind_t &Kind = ParamDescs[I].kind;
    const int &Size = ParamDescs[I].info;
    if (Kind == detail::kernel_param_kind_t::kind_accessor) {
      // For args kind of accessor Size is information about accessor.
      // The first 11 bits of Size encodes the accessor target.
      const access::target AccTarget =
          static_cast<access::target>(Size & AccessTargetMask);
      if ((AccTarget == access::target::device ||
          AccTarget == access::target::constant_buffer) ||
          (AccTarget == access::target::image ||
          AccTarget == access::target::image_array)) {
        detail::AccessorBaseHost *AccBase =
            static_cast<detail::AccessorBaseHost *>(Ptr);
        Ptr = detail::getSyclObjImpl(*AccBase).get();
      } else if (AccTarget == access::target::local) {
        detail::LocalAccessorBaseHost *LocalAccBase =
            static_cast<detail::LocalAccessorBaseHost *>(Ptr);
        Ptr = detail::getSyclObjImpl(*LocalAccBase).get();
      }
    }
    processArg(Ptr, Kind, Size, I, IndexShift, IsKernelCreatedFromSource,
              IsESIMD);
  }
}



template <typename KernelName, typename KernelType, int Dims,
          typename LambdaArgType>
void queue::StoreLambda(KernelType KernelFunc) {
  constexpr bool IsCallableWithKernelHandler =
      detail::KernelLambdaHasKernelHandlerArgT<KernelType,
                                                LambdaArgType>::value;

  impl->MHostKernel = std::make_unique<
      detail::HostKernel<KernelType, LambdaArgType, Dims>>(KernelFunc);

  constexpr bool KernelHasName =
      detail::getKernelName<KernelName>() != nullptr &&
      detail::getKernelName<KernelName>()[0] != '\0';

  // Some host compilers may have different captures from Clang. Currently
  // there is no stable way of handling this when extracting the captures, so
  // a static assert is made to fail for incompatible kernel lambdas.

  // TODO remove the ifdef once the kernel size builtin is supported.
#ifdef __INTEL_SYCL_USE_INTEGRATION_HEADERS
  static_assert(
      !KernelHasName ||
          sizeof(KernelFunc) == detail::getKernelSize<KernelName>(),
      "Unexpected kernel lambda size. This can be caused by an "
      "external host compiler producing a lambda with an "
      "unexpected layout. This is a limitation of the compiler."
      "In many cases the difference is related to capturing constexpr "
      "variables. In such cases removing constexpr specifier aligns the "
      "captures between the host compiler and the device compiler."
      "\n"
      "In case of MSVC, passing "
      "-fsycl-host-compiler-options='/std:c++latest' "
      "might also help.");
#endif
  // Empty name indicates that the compilation happens without integration
  // header, so don't perform things that require it.
  if (KernelHasName) {
    // TODO support ESIMD in no-integration-header case too.
    clearArgs();
    extractArgsAndReqsFromLambda(impl->MHostKernel->getPtr(),
                                  detail::getKernelParamDescs<KernelName>(),
                                  detail::isKernelESIMD<KernelName>());
    impl->MKernelName = detail::getKernelName<KernelName>();
  } else {
    // In case w/o the integration header it is necessary to process
    // accessors from the list(which are associated with this handler) as
    // arguments. We must copy the associated accessors as they are checked
    // later during finalize.
    setArgsToAssociatedAccessors();
  }

  // If the kernel lambda is callable with a kernel_handler argument, manifest
  // the associated kernel handler.
  if (IsCallableWithKernelHandler) {
    getOrInsertHandlerKernelBundle(/*Insert=*/true);
  }
}

void queue::setType(sycl::detail::CGType Type) { impl->MCGType = Type; }

void queue::saveCodeLoc(detail::code_location CodeLoc, bool IsTopCodeLoc) {
  impl->MCodeLoc = CodeLoc;
  impl->MIsTopCodeLoc = IsTopCodeLoc;
}

std::shared_ptr<ext::oneapi::experimental::detail::graph_impl>
queue::getCommandGraphFromHandler() const {
  if (impl->MGraphFromHandler) {
    return impl->MGraphFromHandler;
  }

  return impl->getCommandGraph();
  /*
  // We should never reach here. MGraph and MQueue can not be null
  // simultaneously.
  return nullptr;
  */
}

void queue::depends_on(event Event) {
  auto EventImpl = detail::getSyclObjImpl(Event);
  depends_on(EventImpl);
}

void queue::depends_on(const detail::EventImplPtr &EventImpl) {
  if (!EventImpl)
    return;
  if (EventImpl->isDiscarded()) {
    throw sycl::exception(make_error_code(errc::invalid),
                          "Queue operation cannot depend on discarded event.");
  }

  auto EventGraph = EventImpl->getCommandGraph();
  if (EventGraph) {
    auto QueueGraph = impl->getCommandGraph();

    if (EventGraph->getContext() != get_context()) {
      throw sycl::exception(
          make_error_code(errc::invalid),
          "Cannot submit to a queue with a dependency from a graph that is "
          "associated with a different context.");
    }

    if (EventGraph->getDevice() != get_device()) {
      throw sycl::exception(
          make_error_code(errc::invalid),
          "Cannot submit to a queue with a dependency from a graph that is "
          "associated with a different device.");
    }

    if (QueueGraph && QueueGraph != EventGraph) {
      throw sycl::exception(sycl::make_error_code(errc::invalid),
                            "Cannot submit to a recording queue with a "
                            "dependency from a different graph.");
    }

    // If the event dependency has a graph, that means that the queue that
    // created it was in recording mode. If the current queue is not recording,
    // we need to set it to recording (implements the transitive queue recording
    // feature).
    if (!QueueGraph) {
      EventGraph->beginRecording(impl);
    }
  }

  if (auto Graph = getCommandGraphFromHandler(); Graph) {
    if (EventGraph == nullptr) {
      throw sycl::exception(
          make_error_code(errc::invalid),
          "Graph nodes cannot depend on events from outside the graph.");
    }
    if (EventGraph != Graph) {
      throw sycl::exception(
          make_error_code(errc::invalid),
          "Graph nodes cannot depend on events from another graph.");
    }
  }
  impl->CGDataFromHandler.MEvents.push_back(EventImpl);
}

void queue::depends_on(const std::vector<event> &Events) {
  for (const event &Event : Events) {
    depends_on(Event);
  }
}

void queue::depends_on(const std::vector<detail::EventImplPtr> &Events) {
  for (const detail::EventImplPtr &Event : Events) {
    depends_on(Event);
  }
}

/*
std::tuple<const RTDeviceBinaryImage *, ur_program_handle_t>
queue::retrieveKernelBinary(const QueueImplPtr &Queue, const char *KernelName,
                    CGExecKernel *KernelCG = nullptr) {
  bool isNvidia =
      Queue->getDeviceImplPtr()->getBackend() == backend::ext_oneapi_cuda;
  bool isHIP =
      Queue->getDeviceImplPtr()->getBackend() == backend::ext_oneapi_hip;
  if (isNvidia || isHIP) {
    auto KernelID = ProgramManager::getInstance().getSYCLKernelID(KernelName);
    std::vector<kernel_id> KernelIds{KernelID};
    auto DeviceImages =
        ProgramManager::getInstance().getRawDeviceImages(KernelIds);
    auto DeviceImage = std::find_if(
        DeviceImages.begin(), DeviceImages.end(),
        [isNvidia](RTDeviceBinaryImage *DI) {
          const std::string &TargetSpec = isNvidia ? std::string("llvm_nvptx64")
                                                  : std::string("llvm_amdgcn");
          return DI->getFormat() == SYCL_DEVICE_BINARY_TYPE_LLVMIR_BITCODE &&
                DI->getRawData().DeviceTargetSpec == TargetSpec;
        });
    if (DeviceImage == DeviceImages.end()) {
      return {nullptr, nullptr};
    }
    auto ContextImpl = Queue->getContextImplPtr();
    auto Context = detail::createSyclObjFromImpl<context>(ContextImpl);
    auto DeviceImpl = Queue->getDeviceImplPtr();
    auto Device = detail::createSyclObjFromImpl<device>(DeviceImpl);
    ur_program_handle_t Program =
        detail::ProgramManager::getInstance().createURProgram(
            **DeviceImage, Context, {std::move(Device)});
    return {*DeviceImage, Program};
  }

  const RTDeviceBinaryImage *DeviceImage = nullptr;
  ur_program_handle_t Program = nullptr;
  if (KernelCG->getKernelBundle() != nullptr) {
    // Retrieve the device image from the kernel bundle.
    auto KernelBundle = KernelCG->getKernelBundle();
    kernel_id KernelID =
        detail::ProgramManager::getInstance().getSYCLKernelID(KernelName);

    auto SyclKernel = detail::getSyclObjImpl(
        KernelBundle->get_kernel(KernelID, KernelBundle));

    DeviceImage = SyclKernel->getDeviceImage()->get_bin_image_ref();
    Program = SyclKernel->getDeviceImage()->get_ur_program_ref();
  } else if (KernelCG->MSyclKernel != nullptr) {
    DeviceImage = KernelCG->MSyclKernel->getDeviceImage()->get_bin_image_ref();
    Program = KernelCG->MSyclKernel->getDeviceImage()->get_ur_program_ref();
  } else {
    auto ContextImpl = Queue->getContextImplPtr();
    auto Context = detail::createSyclObjFromImpl<context>(ContextImpl);
    auto DeviceImpl = Queue->getDeviceImplPtr();
    auto Device = detail::createSyclObjFromImpl<device>(DeviceImpl);
    DeviceImage = &detail::ProgramManager::getInstance().getDeviceImage(
        KernelName, Context, Device);
    Program = detail::ProgramManager::getInstance().createURProgram(
        *DeviceImage, Context, {std::move(Device)});
  }
  return {DeviceImage, Program};
}
*/

event queue::finalizeFromHandler() {
  if (impl->MIsFinalized)
    return impl->MLastEvent;
  impl->MIsFinalized = true;

  // According to 4.7.6.9 of SYCL2020 spec, if a placeholder accessor is passed
  // to a command without being bound to a command group, an exception should
  // be thrown.
  {
    for (const auto &arg : impl->MArgs) {
      if (arg.MType != detail::kernel_param_kind_t::kind_accessor)
        continue;

      detail::Requirement *AccImpl =
          static_cast<detail::Requirement *>(arg.MPtr);
      if (AccImpl->MIsPlaceH) {
        auto It = std::find(impl->CGDataFromHandler.MRequirements.begin(),
                            impl->CGDataFromHandler.MRequirements.end(), AccImpl);
        if (It == impl->CGDataFromHandler.MRequirements.end())
          throw sycl::exception(make_error_code(errc::kernel_argument),
                                "placeholder accessor must be bound by calling "
                                "handler::require() before it can be used.");

        // Check associated accessors
        bool AccFound = false;
        for (detail::ArgDesc &Acc : impl->MAssociatedAccesors) {
          if (Acc.MType == detail::kernel_param_kind_t::kind_accessor &&
              static_cast<detail::Requirement *>(Acc.MPtr) == AccImpl) {
            AccFound = true;
            break;
          }
        }

        if (!AccFound) {
          throw sycl::exception(make_error_code(errc::kernel_argument),
                                "placeholder accessor must be bound by calling "
                                "handler::require() before it can be used.");
        }
      }
    }
  }

  const auto &type = getType();
  if (type == detail::CGType::Kernel) {
    // If there were uses of set_specialization_constant build the kernel_bundle
    std::shared_ptr<detail::kernel_bundle_impl> KernelBundleImpPtr =
        getOrInsertHandlerKernelBundle(/*Insert=*/false);
    if (KernelBundleImpPtr) {
      volatile int temp = 123;
      temp++;
    }

    if (!impl->MGraphFromHandler && !impl->MSubgraphNode &&
        !impl->getCommandGraph() && !impl->CGDataFromHandler.MRequirements.size() &&
        !impl->MStreamStorage.size() &&
        (!impl->CGDataFromHandler.MEvents.size() ||
        (impl->isInOrder() &&
          detail::Scheduler::areEventsSafeForSchedulerBypass(
              impl->CGDataFromHandler.MEvents, impl->getContextImplPtr())))) {
      // if user does not add a new dependency to the dependency graph, i.e.
      // the graph is not changed, then this faster path is used to submit
      // kernel bypassing scheduler and avoiding CommandGroup, Command objects
      // creation.
      std::vector<ur_event_handle_t> RawEvents =
        detail::Command::getUrEvents(impl->CGDataFromHandler.MEvents, impl, false);
      detail::EventImplPtr NewEvent;
/*
#ifdef XPTI_ENABLE_INSTRUMENTATION
      // uint32_t StreamID, uint64_t InstanceID, xpti_td* TraceEvent,
      int32_t StreamID = xptiRegisterStream(detail::SYCL_STREAM_NAME);
      auto [CmdTraceEvent, InstanceID] = emitKernelInstrumentationData(
          StreamID, MKernel, MCodeLoc, impl->MIsTopCodeLoc, MKernelName.c_str(),
          MQueue, impl->MNDRDesc, KernelBundleImpPtr, impl->MArgs);
      auto EnqueueKernel = [&, CmdTraceEvent = CmdTraceEvent,
                            InstanceID = InstanceID]() {
#else
*/
      auto EnqueueKernel = [&]() {
//#endif
/*
#ifdef XPTI_ENABLE_INSTRUMENTATION
        detail::emitInstrumentationGeneral(StreamID, InstanceID, CmdTraceEvent,
                                          xpti::trace_task_begin, nullptr);
#endif
*/
        const detail::RTDeviceBinaryImage *BinImage = nullptr;
        if (detail::SYCLConfig<detail::SYCL_JIT_AMDGCN_PTX_KERNELS>::get()) {
          std::tie(BinImage, std::ignore) =
              retrieveKernelBinary(impl, impl->MKernelName.c_str());
          assert(BinImage && "Failed to obtain a binary image.");
        }

        enqueueImpKernel(impl, impl->MNDRDesc, impl->MArgs,
                        KernelBundleImpPtr, impl->MKernel, impl->MKernelName.c_str(),
                        RawEvents, NewEvent, nullptr, impl->MKernelCacheConfig,
                        impl->MKernelIsCooperative,
                        impl->MKernelUsesClusterLaunch,
                        impl->MKernelWorkGroupMemorySize, BinImage);
/*
#ifdef XPTI_ENABLE_INSTRUMENTATION
        // Emit signal only when event is created
        if (NewEvent != nullptr) {
          detail::emitInstrumentationGeneral(
              StreamID, InstanceID, CmdTraceEvent, xpti::trace_signal,
              static_cast<const void *>(NewEvent->getHandle()));
        }
        detail::emitInstrumentationGeneral(StreamID, InstanceID, CmdTraceEvent,
                                          xpti::trace_task_end, nullptr);
#endif
*/
      };

      bool DiscardEvent = (impl->MDiscardEvents || !impl->MEventNeeded) &&
                          impl->supportsDiscardingPiEvents();
      if (DiscardEvent) {
        // Kernel only uses assert if it's non interop one
        bool KernelUsesAssert =
            !(impl->MKernel && impl->MKernel->isInterop()) &&
            detail::ProgramManager::getInstance().kernelUsesAssert(
                impl->MKernelName.c_str());
        DiscardEvent = !KernelUsesAssert;
      }

      if (DiscardEvent) {
        EnqueueKernel();
        auto EventImpl = std::make_shared<detail::event_impl>(
            detail::event_impl::HES_Discarded);
        impl->MLastEvent = detail::createSyclObjFromImpl<event>(EventImpl);
      } else {
        NewEvent = std::make_shared<detail::event_impl>(impl);
        NewEvent->setWorkerQueue(impl);
        NewEvent->setContextImpl(impl->getContextImplPtr());
        NewEvent->setStateIncomplete();
        NewEvent->setSubmissionTime();

        EnqueueKernel();
        if (NewEvent->isHost() || NewEvent->getHandle() == nullptr)
          NewEvent->setComplete();
        NewEvent->setEnqueued();

        impl->MLastEvent = detail::createSyclObjFromImpl<event>(NewEvent);
      }
      return impl->MLastEvent;
    }
  }

  return impl->MLastEvent;
}

void queue::finalizeHandlerInQueue(event &EventRet) {
  if (impl->MIsInorder) {
    // Accessing and changing of an event isn't atomic operation.
    // Hence, here is the lock for thread-safety.
    std::lock_guard<std::mutex> Lock{impl->MMutex};

    auto &EventToBuildDeps = impl->MGraph.expired() ? impl->MDefaultGraphDeps.LastEventPtr
                                              : impl->MExtGraphDeps.LastEventPtr;

    // This dependency is needed for the following purposes:
    //    - host tasks are handled by the runtime and cannot be implicitly
    //    synchronized by the backend.
    //    - to prevent the 2nd kernel enqueue when the 1st kernel is blocked
    //    by a host task. This dependency allows to build the enqueue order in
    //    the RT but will not be passed to the backend. See getPIEvents in
    //    Command.
    // TODO check if this is executed
    if (EventToBuildDeps) {
      // In the case where the last event was discarded and we are to run a
      // host_task, we insert a barrier into the queue and use the resulting
      // event as the dependency for the host_task.
      // Note that host_task events can never be discarded, so this will not
      // insert barriers between host_task enqueues.
      /*
      if (EventToBuildDeps->isDiscarded() &&
          MCGType == CGType::CodeplayHostTask)
        EventToBuildDeps = insertHelperBarrier(Handler);
      */

      if (!EventToBuildDeps->isDiscarded())
        depends_on(EventToBuildDeps);
    }

    // If there is an external event set, add it as a dependency and clear it.
    // We do not need to hold the lock as MLastEventMtx will ensure the last
    // event reflects the corresponding external event dependence as well.
    std::optional<event> ExternalEvent = impl->popExternalEvent();
    if (ExternalEvent)
      depends_on(*ExternalEvent);

    EventRet = finalizeFromHandler();
    EventToBuildDeps = getSyclObjImpl(EventRet);
  } else {
    const detail::CGType Type = impl->MCGType;
    std::lock_guard<std::mutex> Lock{impl->MMutex};
    // The following code supports barrier synchronization if host task is
    // involved in the scenario. Native barriers cannot handle host task
    // dependency so in the case where some commands were not enqueued
    // (blocked), we track them to prevent barrier from being enqueued
    // earlier.
    {
      std::lock_guard<std::mutex> RequestLock(impl->MMissedCleanupRequestsMtx);
      for (auto &UpdatedGraph : impl->MMissedCleanupRequests)
        impl->doUnenqueuedCommandCleanup(UpdatedGraph);
      impl->MMissedCleanupRequests.clear();
    }
    auto &Deps = impl->MGraph.expired() ? impl->MDefaultGraphDeps : impl->MExtGraphDeps;
    if (Type == detail::CGType::Barrier && !Deps.UnenqueuedCmdEvents.empty()) {
      depends_on(Deps.UnenqueuedCmdEvents);
    }
    if (Deps.LastBarrier && (Type == detail::CGType::CodeplayHostTask ||
                              (!Deps.LastBarrier->isEnqueued())))
      depends_on(Deps.LastBarrier);

    EventRet = finalizeFromHandler();
    detail::EventImplPtr EventRetImpl = getSyclObjImpl(EventRet);
    if (Type == detail::CGType::CodeplayHostTask)
      Deps.UnenqueuedCmdEvents.push_back(EventRetImpl);
    else if (Type == detail::CGType::Barrier || Type == detail::CGType::BarrierWaitlist) {
      Deps.LastBarrier = EventRetImpl;
      Deps.UnenqueuedCmdEvents.clear();
    } else if (!EventRetImpl->isEnqueued()) {
      Deps.UnenqueuedCmdEvents.push_back(EventRetImpl);
    }
  }
}

template <typename KernelName = detail::auto_name>
void queue::single_task_from_handler(type_erased_no_handler &t) {
  t();
}

template <typename KernelName, typename KernelType>
void queue::single_task_lambda_impl(_KERNELFUNCPARAM(KernelFunc)) {
  // from queue::submit_with_event
  detail::SubmissionInfo SubmitInfo{};

  // from queue_impl::submit_impl
  const detail::code_location &CodeLoc = detail::code_location::current();
  saveCodeLoc(CodeLoc, true);

  // from handler::single_task_lambda_impl

  // TODO: Properties may change the kernel function, so in order to avoid
  //       conflicts they should be included in the name.
  using NameT =
      typename detail::get_kernel_name_t<KernelName, KernelType>::name;

  kernel_single_task_wrapper<NameT, KernelType>(KernelFunc);
#ifndef __SYCL_DEVICE_ONLY__
  throwIfActionIsCreated();
  throwOnKernelParameterMisuse<KernelName, KernelType>();
  verifyUsedKernelBundleInternal(
      detail::string_view{detail::getKernelName<NameT>()});
  // No need to check if range is out of INT_MAX limits as it's compile-time
  // known constant.
  setNDRangeDescriptor(range<1>{1});
  //processProperties<detail::isKernelESIMD<NameT>(), PropertiesT>(Props);
  StoreLambda<NameT, KernelType, /*Dims*/ 1, void>(KernelFunc);
  setType(detail::CGType::Kernel);
#endif

  // from queue_impl::submit_impl

  const detail::CGType Type = impl->MCGType;
  event Event = detail::createSyclObjFromImpl<event>(
      std::make_shared<detail::event_impl>());
  std::vector<detail::StreamImplPtr> Streams;
  if (Type == detail::CGType::Kernel)
    Streams = std::move(impl->MStreamStorage);

  impl->MEventModeFromHandler = SubmitInfo.EventMode();

  if (SubmitInfo.PostProcessorFunc()) {
    auto &PostProcess = *SubmitInfo.PostProcessorFunc();

    // This code will not be executed
    /*
    bool IsKernel = Type == CGType::Kernel;
    bool KernelUsesAssert = false;

    if (IsKernel)
      // Kernel only uses assert if it's non interop one
      KernelUsesAssert = !(Handler.MKernel && Handler.MKernel->isInterop()) &&
                        ProgramManager::getInstance().kernelUsesAssert(
                            Handler.MKernelName.c_str());
    finalizeHandler(Handler, Event);

    PostProcess(IsKernel, KernelUsesAssert, Event);
    */
  } else
    finalizeHandlerInQueue(Event);

  impl->addEvent(Event);
}

/*
template <typename KernelName = detail::auto_name, typename KernelType>
void queue::single_task_from_handler(_KERNELFUNCPARAM(KernelFunc)) {
  single_task_lambda_impl<KernelName>(
      ext::oneapi::experimental::empty_properties_t{}, KernelFunc);
}
*/

} // namespace _V1
} // namespace sycl

size_t std::hash<sycl::queue>::operator()(const sycl::queue &Q) const {
  // Compared to using the impl pointer, the unique ID helps avoid hash
  // collisions with previously destroyed queues.
  return std::hash<unsigned long long>()(
      sycl::detail::getSyclObjImpl(Q)->getQueueID());
}

