// Copyright 2015 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

//! Bindings for jemalloc as an allocator
//!
//! This crate provides bindings to jemalloc as a memory allocator for Rust.
//! This crate mainly exports, one type, `Jemalloc`, which implements the
//! `GlobalAlloc` trait and optionally the `Alloc` trait,
//! and is suitable both as a memory allocator and as a global allocator.

#![cfg_attr(feature = "alloc_trait", feature(allocator_api))]
// TODO: rename the following lint on next minor bump
#![allow(renamed_and_removed_lints)]
#![deny(missing_docs, broken_intra_doc_links)]
#![no_std]

#[cfg(feature = "alloc_trait")]
use core::alloc::{Alloc, AllocErr, CannotReallocInPlace, Excess};
#[cfg(feature = "alloc_trait")]
use core::ptr::NonNull;
use core::{
    alloc::{GlobalAlloc, Layout},
    cmp,
    hint::assert_unchecked,
};

use crate::ffi::{MALLOCX_ALIGN, MALLOCX_ZERO};
use libc::c_void;

// This constant equals _Alignof(max_align_t) and is platform-specific. It
// contains the _maximum_ alignment that the memory allocations returned by the
// C standard library memory allocation APIs (e.g. `malloc`) are guaranteed to
// have.
//
// The memory allocation APIs are required to return memory that can fit any
// object whose fundamental aligment is <= _Alignof(max_align_t).
//
// In C, there are no ZSTs, and the size of all types is a multiple of their
// alignment (size >= align). So for allocations with size <=
// _Alignof(max_align_t), the malloc-APIs return memory whose alignment is
// either the requested size if its a power-of-two, or the next smaller
// power-of-two.
#[cfg(any(target_arch = "arm", target_arch = "mips", target_arch = "powerpc"))]
const QUANTUM: usize = 8;
#[cfg(any(
    target_arch = "x86",
    target_arch = "x86_64",
    target_arch = "aarch64",
    target_arch = "powerpc64",
    target_arch = "loongarch64",
    target_arch = "mips64",
    target_arch = "riscv64",
    target_arch = "s390x",
    target_arch = "sparc64"
))]
const QUANTUM: usize = 16;

#[inline]
unsafe fn adjust_layout(layout: Layout) -> Layout {
    assert_unchecked(layout.align() > 0);
    let align = cmp::max(layout.align(), QUANTUM);
    debug_assert!(align >= size_of::<c_void>(), "alignment too small");
    debug_assert!(align.count_ones() == 1, "alignment not a pow2");

    assert_unchecked(layout.size() > 0);
    let size = cmp::max(layout.size(), QUANTUM);
    debug_assert!(size >= size_of::<c_void>(), "size too small");
    debug_assert!(size >= align, "allocating a fragment");

    Layout::from_size_align_unchecked(size, align)
}

/// Handle to the jemalloc allocator
///
/// This type implements the `GlobalAllocAlloc` trait, allowing usage a global allocator.
///
/// When the `alloc_trait` feature of this crate is enabled, it also implements the `Alloc` trait,
/// allowing usage in collections.
#[derive(Copy, Clone, Default, Debug)]
pub struct Jemalloc;

unsafe impl GlobalAlloc for Jemalloc {
    #[inline]
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        let layout = adjust_layout(layout);
        let flags = MALLOCX_ALIGN(layout.align());
        debug_assert!(
            ffi::nallocx(layout.size(), flags) >= layout.size(),
            "alloc: nallocx() reported failure"
        );

        let ptr = ffi::mallocx(layout.size(), flags);
        debug_assert!(
            ffi::sallocx(ptr, flags) >= layout.size(),
            "alloc: sallocx() size mismatch"
        );

        ptr as *mut u8
    }

    #[inline]
    unsafe fn alloc_zeroed(&self, layout: Layout) -> *mut u8 {
        let layout = adjust_layout(layout);
        let flags = MALLOCX_ALIGN(layout.align()) | MALLOCX_ZERO;
        debug_assert!(
            ffi::nallocx(layout.size(), flags) >= layout.size(),
            "alloc_zeroed: nallocx() reported failure"
        );

        let ptr = ffi::mallocx(layout.size(), flags);
        debug_assert!(
            ffi::sallocx(ptr, flags) >= layout.size(),
            "alloc_zeroed: sallocx() size mismatch"
        );

        ptr as *mut u8
    }

    #[inline]
    unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
        let layout = Layout::from_size_align_unchecked(new_size, layout.align());
        let layout = adjust_layout(layout);
        let flags = MALLOCX_ALIGN(layout.align());
        debug_assert!(
            ffi::nallocx(layout.size(), flags) >= layout.size(),
            "realloc: nallocx() reported failure"
        );

        let ptr = ffi::rallocx(ptr as *mut c_void, layout.size(), flags);
        debug_assert!(
            ffi::sallocx(ptr, flags) >= layout.size(),
            "reelloc: sallocx() size mismatch"
        );

        ptr as *mut u8
    }

    #[inline]
    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        assert_unchecked(!ptr.is_null());
        let ptr = ptr as *mut c_void;
        let layout = adjust_layout(layout);
        let flags = MALLOCX_ALIGN(layout.align());
        debug_assert!(
            ffi::sallocx(ptr, flags) >= layout.size(),
            "dealloc: sallocx() size mismatch"
        );

        ffi::sdallocx(ptr, layout.size(), flags)
    }
}

#[cfg(feature = "alloc_trait")]
unsafe impl Alloc for Jemalloc {
    #[inline]
    unsafe fn alloc(&mut self, layout: Layout) -> Result<NonNull<u8>, AllocErr> {
        NonNull::new(GlobalAlloc::alloc(self, layout)).ok_or(AllocErr)
    }

    #[inline]
    unsafe fn alloc_zeroed(&mut self, layout: Layout) -> Result<NonNull<u8>, AllocErr> {
        NonNull::new(GlobalAlloc::alloc_zeroed(self, layout)).ok_or(AllocErr)
    }

    #[inline]
    unsafe fn dealloc(&mut self, ptr: NonNull<u8>, layout: Layout) {
        GlobalAlloc::dealloc(self, ptr.as_ptr(), layout)
    }

    #[inline]
    unsafe fn realloc(
        &mut self,
        ptr: NonNull<u8>,
        layout: Layout,
        new_size: usize,
    ) -> Result<NonNull<u8>, AllocErr> {
        NonNull::new(GlobalAlloc::realloc(self, ptr.as_ptr(), layout, new_size)).ok_or(AllocErr)
    }

    #[inline]
    unsafe fn alloc_excess(&mut self, layout: Layout) -> Result<Excess, AllocErr> {
        let flags = layout_to_flags(layout.align(), layout.size());
        let ptr = ffi::mallocx(layout.size(), flags);
        if let Some(nonnull) = NonNull::new(ptr as *mut u8) {
            let excess = ffi::nallocx(layout.size(), flags);
            Ok(Excess(nonnull, excess))
        } else {
            Err(AllocErr)
        }
    }

    #[inline]
    unsafe fn realloc_excess(
        &mut self,
        ptr: NonNull<u8>,
        layout: Layout,
        new_size: usize,
    ) -> Result<Excess, AllocErr> {
        let flags = layout_to_flags(layout.align(), new_size);
        let ptr = ffi::rallocx(ptr.cast().as_ptr(), new_size, flags);
        if let Some(nonnull) = NonNull::new(ptr as *mut u8) {
            let excess = ffi::nallocx(new_size, flags);
            Ok(Excess(nonnull, excess))
        } else {
            Err(AllocErr)
        }
    }

    #[inline]
    fn usable_size(&self, layout: &Layout) -> (usize, usize) {
        let flags = layout_to_flags(layout.align(), layout.size());
        unsafe {
            let max = ffi::nallocx(layout.size(), flags);
            (layout.size(), max)
        }
    }

    #[inline]
    unsafe fn grow_in_place(
        &mut self,
        ptr: NonNull<u8>,
        layout: Layout,
        new_size: usize,
    ) -> Result<(), CannotReallocInPlace> {
        let flags = layout_to_flags(layout.align(), new_size);
        let usable_size = ffi::xallocx(ptr.cast().as_ptr(), new_size, 0, flags);
        if usable_size >= new_size {
            Ok(())
        } else {
            // `xallocx` returns a size smaller than the requested one to
            // indicate that the allocation could not be grown in place
            //
            // the old allocation remains unaltered
            Err(CannotReallocInPlace)
        }
    }

    #[inline]
    unsafe fn shrink_in_place(
        &mut self,
        ptr: NonNull<u8>,
        layout: Layout,
        new_size: usize,
    ) -> Result<(), CannotReallocInPlace> {
        if new_size == layout.size() {
            return Ok(());
        }
        let flags = layout_to_flags(layout.align(), new_size);
        let usable_size = ffi::xallocx(ptr.cast().as_ptr(), new_size, 0, flags);

        if usable_size < layout.size() {
            // If `usable_size` is smaller than the original size, the
            // size-class of the allocation was shrunk to the size-class of
            // `new_size`, and it is safe to deallocate the allocation with
            // `new_size`:
            Ok(())
        } else if usable_size == ffi::nallocx(new_size, flags) {
            // If the allocation was not shrunk and the size class of `new_size`
            // is the same as the size-class of `layout.size()`, then the
            // allocation can be properly deallocated using `new_size` (and also
            // using `layout.size()` because the allocation did not change)

            // note: when the allocation is not shrunk, `xallocx` returns the
            // usable size of the original allocation, which in this case matches
            // that of the requested allocation:
            debug_assert_eq!(
                ffi::nallocx(new_size, flags),
                ffi::nallocx(layout.size(), flags)
            );
            Ok(())
        } else {
            // If the allocation was not shrunk, but the size-class of
            // `new_size` is not the same as that of the original allocation,
            // then shrinking the allocation failed:
            Err(CannotReallocInPlace)
        }
    }
}

/// Return the usable size of the allocation pointed to by ptr.
///
/// The return value may be larger than the size that was requested during allocation.
/// This function is not a mechanism for in-place `realloc()`;
/// rather it is provided solely as a tool for introspection purposes.
/// Any discrepancy between the requested allocation size
/// and the size reported by this function should not be depended on,
/// since such behavior is entirely implementation-dependent.
///
/// # Safety
///
/// `ptr` must have been allocated by `Jemalloc` and must not have been freed yet.
pub unsafe fn usable_size<T>(ptr: *const T) -> usize {
    ffi::malloc_usable_size(ptr as *const c_void)
}

/// Raw bindings to jemalloc
mod ffi {
    pub use tikv_jemalloc_sys::*;
}
