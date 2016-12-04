; ModuleID = 'cu_builtins.bc'
target datalayout = "e-i64:64-v16:16-v32:32-n16:32:64"
target triple = "nvptx64-nvidia-cuda"

define i64 @amp_get_global_size(i32) #0 {
  %2 = alloca i64, align 8
  %3 = alloca i32, align 4
  store i32 %0, i32* %3, align 4
  %4 = load i32, i32* %3, align 4
  %5 = icmp eq i32 %4, 0
  br i1 %5, label %6, label %11

; <label>:6:                                      ; preds = %1
  %7 = call i32 @llvm.nvvm.read.ptx.sreg.nctaid.x()
  %8 = call i32 @llvm.nvvm.read.ptx.sreg.ntid.x()
  %9 = mul nsw i32 %7, %8
  %10 = sext i32 %9 to i64
  store i64 %10, i64* %2, align 8
  br label %24

; <label>:11:                                     ; preds = %1
  %12 = load i32, i32* %3, align 4
  %13 = icmp eq i32 %12, 1
  br i1 %13, label %14, label %19

; <label>:14:                                     ; preds = %11
  %15 = call i32 @llvm.nvvm.read.ptx.sreg.nctaid.y()
  %16 = call i32 @llvm.nvvm.read.ptx.sreg.ntid.y()
  %17 = mul nsw i32 %15, %16
  %18 = sext i32 %17 to i64
  store i64 %18, i64* %2, align 8
  br label %24

; <label>:19:                                     ; preds = %11
  %20 = call i32 @llvm.nvvm.read.ptx.sreg.nctaid.z()
  %21 = call i32 @llvm.nvvm.read.ptx.sreg.ntid.z()
  %22 = mul nsw i32 %20, %21
  %23 = sext i32 %22 to i64
  store i64 %23, i64* %2, align 8
  br label %24

; <label>:24:                                     ; preds = %19, %14, %6
  %25 = load i64, i64* %2, align 8
  ret i64 %25
}

; Function Attrs: nounwind readnone
declare i32 @llvm.nvvm.read.ptx.sreg.nctaid.x() #1

; Function Attrs: nounwind readnone
declare i32 @llvm.nvvm.read.ptx.sreg.ntid.x() #1

; Function Attrs: nounwind readnone
declare i32 @llvm.nvvm.read.ptx.sreg.nctaid.y() #1

; Function Attrs: nounwind readnone
declare i32 @llvm.nvvm.read.ptx.sreg.ntid.y() #1

; Function Attrs: nounwind readnone
declare i32 @llvm.nvvm.read.ptx.sreg.nctaid.z() #1

; Function Attrs: nounwind readnone
declare i32 @llvm.nvvm.read.ptx.sreg.ntid.z() #1

define i64 @amp_get_global_id(i32) #0 {
  %2 = alloca i64, align 8
  %3 = alloca i32, align 4
  store i32 %0, i32* %3, align 4
  %4 = load i32, i32* %3, align 4
  %5 = icmp eq i32 %4, 0
  br i1 %5, label %6, label %13

; <label>:6:                                      ; preds = %1
  %7 = call i32 @llvm.nvvm.read.ptx.sreg.ctaid.x()
  %8 = call i32 @llvm.nvvm.read.ptx.sreg.ntid.x()
  %9 = mul nsw i32 %7, %8
  %10 = call i32 @llvm.nvvm.read.ptx.sreg.tid.x()
  %11 = add nsw i32 %9, %10
  %12 = sext i32 %11 to i64
  store i64 %12, i64* %2, align 8
  br label %30

; <label>:13:                                     ; preds = %1
  %14 = load i32, i32* %3, align 4
  %15 = icmp eq i32 %14, 1
  br i1 %15, label %16, label %23

; <label>:16:                                     ; preds = %13
  %17 = call i32 @llvm.nvvm.read.ptx.sreg.ctaid.y()
  %18 = call i32 @llvm.nvvm.read.ptx.sreg.ntid.y()
  %19 = mul nsw i32 %17, %18
  %20 = call i32 @llvm.nvvm.read.ptx.sreg.tid.y()
  %21 = add nsw i32 %19, %20
  %22 = sext i32 %21 to i64
  store i64 %22, i64* %2, align 8
  br label %30

; <label>:23:                                     ; preds = %13
  %24 = call i32 @llvm.nvvm.read.ptx.sreg.ctaid.z()
  %25 = call i32 @llvm.nvvm.read.ptx.sreg.ntid.z()
  %26 = mul nsw i32 %24, %25
  %27 = call i32 @llvm.nvvm.read.ptx.sreg.tid.z()
  %28 = add nsw i32 %26, %27
  %29 = sext i32 %28 to i64
  store i64 %29, i64* %2, align 8
  br label %30

; <label>:30:                                     ; preds = %23, %16, %6
  %31 = load i64, i64* %2, align 8
  ret i64 %31
}

; Function Attrs: nounwind readnone
declare i32 @llvm.nvvm.read.ptx.sreg.ctaid.x() #1

; Function Attrs: nounwind readnone
declare i32 @llvm.nvvm.read.ptx.sreg.tid.x() #1

; Function Attrs: nounwind readnone
declare i32 @llvm.nvvm.read.ptx.sreg.ctaid.y() #1

; Function Attrs: nounwind readnone
declare i32 @llvm.nvvm.read.ptx.sreg.tid.y() #1

; Function Attrs: nounwind readnone
declare i32 @llvm.nvvm.read.ptx.sreg.ctaid.z() #1

; Function Attrs: nounwind readnone
declare i32 @llvm.nvvm.read.ptx.sreg.tid.z() #1

define i64 @amp_get_local_size(i32) #0 {
  %2 = alloca i64, align 8
  %3 = alloca i32, align 4
  store i32 %0, i32* %3, align 4
  %4 = load i32, i32* %3, align 4
  %5 = icmp eq i32 %4, 0
  br i1 %5, label %6, label %9

; <label>:6:                                      ; preds = %1
  %7 = call i32 @llvm.nvvm.read.ptx.sreg.ntid.x()
  %8 = sext i32 %7 to i64
  store i64 %8, i64* %2, align 8
  br label %18

; <label>:9:                                      ; preds = %1
  %10 = load i32, i32* %3, align 4
  %11 = icmp eq i32 %10, 1
  br i1 %11, label %12, label %15

; <label>:12:                                     ; preds = %9
  %13 = call i32 @llvm.nvvm.read.ptx.sreg.ntid.y()
  %14 = sext i32 %13 to i64
  store i64 %14, i64* %2, align 8
  br label %18

; <label>:15:                                     ; preds = %9
  %16 = call i32 @llvm.nvvm.read.ptx.sreg.ntid.z()
  %17 = sext i32 %16 to i64
  store i64 %17, i64* %2, align 8
  br label %18

; <label>:18:                                     ; preds = %15, %12, %6
  %19 = load i64, i64* %2, align 8
  ret i64 %19
}

define i64 @amp_get_local_id(i32) #0 {
  %2 = alloca i64, align 8
  %3 = alloca i32, align 4
  store i32 %0, i32* %3, align 4
  %4 = load i32, i32* %3, align 4
  %5 = icmp eq i32 %4, 0
  br i1 %5, label %6, label %9

; <label>:6:                                      ; preds = %1
  %7 = call i32 @llvm.nvvm.read.ptx.sreg.tid.x()
  %8 = sext i32 %7 to i64
  store i64 %8, i64* %2, align 8
  br label %18

; <label>:9:                                      ; preds = %1
  %10 = load i32, i32* %3, align 4
  %11 = icmp eq i32 %10, 1
  br i1 %11, label %12, label %15

; <label>:12:                                     ; preds = %9
  %13 = call i32 @llvm.nvvm.read.ptx.sreg.tid.y()
  %14 = sext i32 %13 to i64
  store i64 %14, i64* %2, align 8
  br label %18

; <label>:15:                                     ; preds = %9
  %16 = call i32 @llvm.nvvm.read.ptx.sreg.tid.z()
  %17 = sext i32 %16 to i64
  store i64 %17, i64* %2, align 8
  br label %18

; <label>:18:                                     ; preds = %15, %12, %6
  %19 = load i64, i64* %2, align 8
  ret i64 %19
}

define i64 @amp_get_num_groups(i32) #0 {
  %2 = alloca i64, align 8
  %3 = alloca i32, align 4
  store i32 %0, i32* %3, align 4
  %4 = load i32, i32* %3, align 4
  %5 = icmp eq i32 %4, 0
  br i1 %5, label %6, label %9

; <label>:6:                                      ; preds = %1
  %7 = call i32 @llvm.nvvm.read.ptx.sreg.nctaid.x()
  %8 = sext i32 %7 to i64
  store i64 %8, i64* %2, align 8
  br label %18

; <label>:9:                                      ; preds = %1
  %10 = load i32, i32* %3, align 4
  %11 = icmp eq i32 %10, 1
  br i1 %11, label %12, label %15

; <label>:12:                                     ; preds = %9
  %13 = call i32 @llvm.nvvm.read.ptx.sreg.nctaid.y()
  %14 = sext i32 %13 to i64
  store i64 %14, i64* %2, align 8
  br label %18

; <label>:15:                                     ; preds = %9
  %16 = call i32 @llvm.nvvm.read.ptx.sreg.nctaid.z()
  %17 = sext i32 %16 to i64
  store i64 %17, i64* %2, align 8
  br label %18

; <label>:18:                                     ; preds = %15, %12, %6
  %19 = load i64, i64* %2, align 8
  ret i64 %19
}

define i64 @amp_get_group_id(i32) #0 {
  %2 = alloca i64, align 8
  %3 = alloca i32, align 4
  store i32 %0, i32* %3, align 4
  %4 = load i32, i32* %3, align 4
  %5 = icmp eq i32 %4, 0
  br i1 %5, label %6, label %9

; <label>:6:                                      ; preds = %1
  %7 = call i32 @llvm.nvvm.read.ptx.sreg.ctaid.x()
  %8 = sext i32 %7 to i64
  store i64 %8, i64* %2, align 8
  br label %18

; <label>:9:                                      ; preds = %1
  %10 = load i32, i32* %3, align 4
  %11 = icmp eq i32 %10, 1
  br i1 %11, label %12, label %15

; <label>:12:                                     ; preds = %9
  %13 = call i32 @llvm.nvvm.read.ptx.sreg.ctaid.y()
  %14 = sext i32 %13 to i64
  store i64 %14, i64* %2, align 8
  br label %18

; <label>:15:                                     ; preds = %9
  %16 = call i32 @llvm.nvvm.read.ptx.sreg.ctaid.z()
  %17 = sext i32 %16 to i64
  store i64 %17, i64* %2, align 8
  br label %18

; <label>:18:                                     ; preds = %15, %12, %6
  %19 = load i64, i64* %2, align 8
  ret i64 %19
}

define void @amp_barrier(i32) #0 {
  %2 = alloca i32, align 4
  store i32 %0, i32* %2, align 4
  call void @llvm.nvvm.barrier0()
  ret void
}

; Function Attrs: convergent nounwind
declare void @llvm.nvvm.barrier0() #2

define i64 @hc_get_grid_size(i32) #0 {
  %2 = alloca i64, align 8
  %3 = alloca i32, align 4
  store i32 %0, i32* %3, align 4
  %4 = load i32, i32* %3, align 4
  %5 = icmp eq i32 %4, 0
  br i1 %5, label %6, label %11

; <label>:6:                                      ; preds = %1
  %7 = call i32 @llvm.nvvm.read.ptx.sreg.nctaid.x()
  %8 = call i32 @llvm.nvvm.read.ptx.sreg.ntid.x()
  %9 = mul nsw i32 %7, %8
  %10 = sext i32 %9 to i64
  store i64 %10, i64* %2, align 8
  br label %24

; <label>:11:                                     ; preds = %1
  %12 = load i32, i32* %3, align 4
  %13 = icmp eq i32 %12, 1
  br i1 %13, label %14, label %19

; <label>:14:                                     ; preds = %11
  %15 = call i32 @llvm.nvvm.read.ptx.sreg.nctaid.y()
  %16 = call i32 @llvm.nvvm.read.ptx.sreg.ntid.y()
  %17 = mul nsw i32 %15, %16
  %18 = sext i32 %17 to i64
  store i64 %18, i64* %2, align 8
  br label %24

; <label>:19:                                     ; preds = %11
  %20 = call i32 @llvm.nvvm.read.ptx.sreg.nctaid.z()
  %21 = call i32 @llvm.nvvm.read.ptx.sreg.ntid.z()
  %22 = mul nsw i32 %20, %21
  %23 = sext i32 %22 to i64
  store i64 %23, i64* %2, align 8
  br label %24

; <label>:24:                                     ; preds = %19, %14, %6
  %25 = load i64, i64* %2, align 8
  ret i64 %25
}

define i64 @hc_get_workitem_absolute_id(i32) #0 {
  %2 = alloca i64, align 8
  %3 = alloca i32, align 4
  store i32 %0, i32* %3, align 4
  %4 = load i32, i32* %3, align 4
  %5 = icmp eq i32 %4, 0
  br i1 %5, label %6, label %13

; <label>:6:                                      ; preds = %1
  %7 = call i32 @llvm.nvvm.read.ptx.sreg.ctaid.x()
  %8 = call i32 @llvm.nvvm.read.ptx.sreg.ntid.x()
  %9 = mul nsw i32 %7, %8
  %10 = call i32 @llvm.nvvm.read.ptx.sreg.tid.x()
  %11 = add nsw i32 %9, %10
  %12 = sext i32 %11 to i64
  store i64 %12, i64* %2, align 8
  br label %30

; <label>:13:                                     ; preds = %1
  %14 = load i32, i32* %3, align 4
  %15 = icmp eq i32 %14, 1
  br i1 %15, label %16, label %23

; <label>:16:                                     ; preds = %13
  %17 = call i32 @llvm.nvvm.read.ptx.sreg.ctaid.y()
  %18 = call i32 @llvm.nvvm.read.ptx.sreg.ntid.y()
  %19 = mul nsw i32 %17, %18
  %20 = call i32 @llvm.nvvm.read.ptx.sreg.tid.y()
  %21 = add nsw i32 %19, %20
  %22 = sext i32 %21 to i64
  store i64 %22, i64* %2, align 8
  br label %30

; <label>:23:                                     ; preds = %13
  %24 = call i32 @llvm.nvvm.read.ptx.sreg.ctaid.z()
  %25 = call i32 @llvm.nvvm.read.ptx.sreg.ntid.z()
  %26 = mul nsw i32 %24, %25
  %27 = call i32 @llvm.nvvm.read.ptx.sreg.tid.z()
  %28 = add nsw i32 %26, %27
  %29 = sext i32 %28 to i64
  store i64 %29, i64* %2, align 8
  br label %30

; <label>:30:                                     ; preds = %23, %16, %6
  %31 = load i64, i64* %2, align 8
  ret i64 %31
}

define i64 @hc_get_group_size(i32) #0 {
  %2 = alloca i64, align 8
  %3 = alloca i32, align 4
  store i32 %0, i32* %3, align 4
  %4 = load i32, i32* %3, align 4
  %5 = icmp eq i32 %4, 0
  br i1 %5, label %6, label %9

; <label>:6:                                      ; preds = %1
  %7 = call i32 @llvm.nvvm.read.ptx.sreg.ntid.x()
  %8 = sext i32 %7 to i64
  store i64 %8, i64* %2, align 8
  br label %18

; <label>:9:                                      ; preds = %1
  %10 = load i32, i32* %3, align 4
  %11 = icmp eq i32 %10, 1
  br i1 %11, label %12, label %15

; <label>:12:                                     ; preds = %9
  %13 = call i32 @llvm.nvvm.read.ptx.sreg.ntid.y()
  %14 = sext i32 %13 to i64
  store i64 %14, i64* %2, align 8
  br label %18

; <label>:15:                                     ; preds = %9
  %16 = call i32 @llvm.nvvm.read.ptx.sreg.ntid.z()
  %17 = sext i32 %16 to i64
  store i64 %17, i64* %2, align 8
  br label %18

; <label>:18:                                     ; preds = %15, %12, %6
  %19 = load i64, i64* %2, align 8
  ret i64 %19
}

define i64 @hc_get_workitem_id(i32) #0 {
  %2 = alloca i64, align 8
  %3 = alloca i32, align 4
  store i32 %0, i32* %3, align 4
  %4 = load i32, i32* %3, align 4
  %5 = icmp eq i32 %4, 0
  br i1 %5, label %6, label %9

; <label>:6:                                      ; preds = %1
  %7 = call i32 @llvm.nvvm.read.ptx.sreg.tid.x()
  %8 = sext i32 %7 to i64
  store i64 %8, i64* %2, align 8
  br label %18

; <label>:9:                                      ; preds = %1
  %10 = load i32, i32* %3, align 4
  %11 = icmp eq i32 %10, 1
  br i1 %11, label %12, label %15

; <label>:12:                                     ; preds = %9
  %13 = call i32 @llvm.nvvm.read.ptx.sreg.tid.y()
  %14 = sext i32 %13 to i64
  store i64 %14, i64* %2, align 8
  br label %18

; <label>:15:                                     ; preds = %9
  %16 = call i32 @llvm.nvvm.read.ptx.sreg.tid.z()
  %17 = sext i32 %16 to i64
  store i64 %17, i64* %2, align 8
  br label %18

; <label>:18:                                     ; preds = %15, %12, %6
  %19 = load i64, i64* %2, align 8
  ret i64 %19
}

define i64 @hc_get_num_groups(i32) #0 {
  %2 = alloca i64, align 8
  %3 = alloca i32, align 4
  store i32 %0, i32* %3, align 4
  %4 = load i32, i32* %3, align 4
  %5 = icmp eq i32 %4, 0
  br i1 %5, label %6, label %9

; <label>:6:                                      ; preds = %1
  %7 = call i32 @llvm.nvvm.read.ptx.sreg.nctaid.x()
  %8 = sext i32 %7 to i64
  store i64 %8, i64* %2, align 8
  br label %18

; <label>:9:                                      ; preds = %1
  %10 = load i32, i32* %3, align 4
  %11 = icmp eq i32 %10, 1
  br i1 %11, label %12, label %15

; <label>:12:                                     ; preds = %9
  %13 = call i32 @llvm.nvvm.read.ptx.sreg.nctaid.y()
  %14 = sext i32 %13 to i64
  store i64 %14, i64* %2, align 8
  br label %18

; <label>:15:                                     ; preds = %9
  %16 = call i32 @llvm.nvvm.read.ptx.sreg.nctaid.z()
  %17 = sext i32 %16 to i64
  store i64 %17, i64* %2, align 8
  br label %18

; <label>:18:                                     ; preds = %15, %12, %6
  %19 = load i64, i64* %2, align 8
  ret i64 %19
}

define i64 @hc_get_group_id(i32) #0 {
  %2 = alloca i64, align 8
  %3 = alloca i32, align 4
  store i32 %0, i32* %3, align 4
  %4 = load i32, i32* %3, align 4
  %5 = icmp eq i32 %4, 0
  br i1 %5, label %6, label %9

; <label>:6:                                      ; preds = %1
  %7 = call i32 @llvm.nvvm.read.ptx.sreg.ctaid.x()
  %8 = sext i32 %7 to i64
  store i64 %8, i64* %2, align 8
  br label %18

; <label>:9:                                      ; preds = %1
  %10 = load i32, i32* %3, align 4
  %11 = icmp eq i32 %10, 1
  br i1 %11, label %12, label %15

; <label>:12:                                     ; preds = %9
  %13 = call i32 @llvm.nvvm.read.ptx.sreg.ctaid.y()
  %14 = sext i32 %13 to i64
  store i64 %14, i64* %2, align 8
  br label %18

; <label>:15:                                     ; preds = %9
  %16 = call i32 @llvm.nvvm.read.ptx.sreg.ctaid.z()
  %17 = sext i32 %16 to i64
  store i64 %17, i64* %2, align 8
  br label %18

; <label>:18:                                     ; preds = %15, %12, %6
  %19 = load i64, i64* %2, align 8
  ret i64 %19
}

define void @hc_barrier(i32) #0 {
  %2 = alloca i32, align 4
  store i32 %0, i32* %2, align 4
  call void @llvm.nvvm.barrier0()
  ret void
}

define float @__hc_acos(float) #0 {
  %2 = alloca float, align 4
  store float %0, float* %2, align 4
  %3 = load float, float* %2, align 4
  %4 = call float @__nv_acosf(float %3)
  ret float %4
}

declare float @__nv_acosf(float) #3

define double @__hc_acos_double(double) #0 {
  %2 = alloca double, align 8
  store double %0, double* %2, align 8
  %3 = load double, double* %2, align 8
  %4 = call double @__nv_acos(double %3)
  ret double %4
}

declare double @__nv_acos(double) #3

define float @__hc_acosh(float) #0 {
  %2 = alloca float, align 4
  store float %0, float* %2, align 4
  %3 = load float, float* %2, align 4
  %4 = call float @__nv_acoshf(float %3)
  ret float %4
}

declare float @__nv_acoshf(float) #3

define double @__hc_acosh_double(double) #0 {
  %2 = alloca double, align 8
  store double %0, double* %2, align 8
  %3 = load double, double* %2, align 8
  %4 = call double @__nv_acosh(double %3)
  ret double %4
}

declare double @__nv_acosh(double) #3

define float @__hc_asin(float) #0 {
  %2 = alloca float, align 4
  store float %0, float* %2, align 4
  %3 = load float, float* %2, align 4
  %4 = call float @__nv_asinf(float %3)
  ret float %4
}

declare float @__nv_asinf(float) #3

define double @__hc_asin_double(double) #0 {
  %2 = alloca double, align 8
  store double %0, double* %2, align 8
  %3 = load double, double* %2, align 8
  %4 = call double @__nv_asin(double %3)
  ret double %4
}

declare double @__nv_asin(double) #3

define float @__hc_asinh(float) #0 {
  %2 = alloca float, align 4
  store float %0, float* %2, align 4
  %3 = load float, float* %2, align 4
  %4 = call float @__nv_asinhf(float %3)
  ret float %4
}

declare float @__nv_asinhf(float) #3

define double @__hc_asinh_double(double) #0 {
  %2 = alloca double, align 8
  store double %0, double* %2, align 8
  %3 = load double, double* %2, align 8
  %4 = call double @__nv_asinh(double %3)
  ret double %4
}

declare double @__nv_asinh(double) #3

define float @__hc_atan(float) #0 {
  %2 = alloca float, align 4
  store float %0, float* %2, align 4
  %3 = load float, float* %2, align 4
  %4 = call float @__nv_atanf(float %3)
  ret float %4
}

declare float @__nv_atanf(float) #3

define double @__hc_atan_double(double) #0 {
  %2 = alloca double, align 8
  store double %0, double* %2, align 8
  %3 = load double, double* %2, align 8
  %4 = call double @__nv_atan(double %3)
  ret double %4
}

declare double @__nv_atan(double) #3

define float @__hc_atanh(float) #0 {
  %2 = alloca float, align 4
  store float %0, float* %2, align 4
  %3 = load float, float* %2, align 4
  %4 = call float @__nv_atanhf(float %3)
  ret float %4
}

declare float @__nv_atanhf(float) #3

define double @__hc_atanh_double(double) #0 {
  %2 = alloca double, align 8
  store double %0, double* %2, align 8
  %3 = load double, double* %2, align 8
  %4 = call double @__nv_atanh(double %3)
  ret double %4
}

declare double @__nv_atanh(double) #3

define float @__hc_atan2(float, float) #0 {
  %3 = alloca float, align 4
  %4 = alloca float, align 4
  store float %0, float* %3, align 4
  store float %1, float* %4, align 4
  %5 = load float, float* %3, align 4
  %6 = load float, float* %4, align 4
  %7 = call float @__nv_atan2f(float %5, float %6)
  ret float %7
}

declare float @__nv_atan2f(float, float) #3

define double @__hc_atan2_double(double, double) #0 {
  %3 = alloca double, align 8
  %4 = alloca double, align 8
  store double %0, double* %3, align 8
  store double %1, double* %4, align 8
  %5 = load double, double* %3, align 8
  %6 = load double, double* %4, align 8
  %7 = call double @__nv_atan2(double %5, double %6)
  ret double %7
}

declare double @__nv_atan2(double, double) #3

define float @__hc_cbrt(float) #0 {
  %2 = alloca float, align 4
  store float %0, float* %2, align 4
  %3 = load float, float* %2, align 4
  %4 = call float @__nv_cbrtf(float %3)
  ret float %4
}

declare float @__nv_cbrtf(float) #3

define double @__hc_cbrt_double(double) #0 {
  %2 = alloca double, align 8
  store double %0, double* %2, align 8
  %3 = load double, double* %2, align 8
  %4 = call double @__nv_cbrt(double %3)
  ret double %4
}

declare double @__nv_cbrt(double) #3

define float @__hc_ceil(float) #0 {
  %2 = alloca float, align 4
  store float %0, float* %2, align 4
  %3 = load float, float* %2, align 4
  %4 = call float @__nv_ceilf(float %3)
  ret float %4
}

declare float @__nv_ceilf(float) #3

define double @__hc_ceil_double(double) #0 {
  %2 = alloca double, align 8
  store double %0, double* %2, align 8
  %3 = load double, double* %2, align 8
  %4 = call double @__nv_ceil(double %3)
  ret double %4
}

declare double @__nv_ceil(double) #3

define float @__hc_copysign(float, float) #0 {
  %3 = alloca float, align 4
  %4 = alloca float, align 4
  store float %0, float* %3, align 4
  store float %1, float* %4, align 4
  %5 = load float, float* %3, align 4
  %6 = load float, float* %4, align 4
  %7 = call float @__nv_copysignf(float %5, float %6)
  ret float %7
}

declare float @__nv_copysignf(float, float) #3

define double @__hc_copysign_double(double, double) #0 {
  %3 = alloca double, align 8
  %4 = alloca double, align 8
  store double %0, double* %3, align 8
  store double %1, double* %4, align 8
  %5 = load double, double* %3, align 8
  %6 = load double, double* %4, align 8
  %7 = call double @__nv_copysign(double %5, double %6)
  ret double %7
}

declare double @__nv_copysign(double, double) #3

define float @__hc_cos(float) #0 {
  %2 = alloca float, align 4
  store float %0, float* %2, align 4
  %3 = load float, float* %2, align 4
  %4 = call float @__nv_cosf(float %3)
  ret float %4
}

declare float @__nv_cosf(float) #3

define double @__hc_cos_double(double) #0 {
  %2 = alloca double, align 8
  store double %0, double* %2, align 8
  %3 = load double, double* %2, align 8
  %4 = call double @__nv_cos(double %3)
  ret double %4
}

declare double @__nv_cos(double) #3

define float @__hc_cosf(float) #0 {
  %2 = alloca float, align 4
  store float %0, float* %2, align 4
  %3 = load float, float* %2, align 4
  %4 = call float @__nv_cosf(float %3)
  ret float %4
}

define float @__hc_cosh(float) #0 {
  %2 = alloca float, align 4
  store float %0, float* %2, align 4
  %3 = load float, float* %2, align 4
  %4 = call float @__nv_coshf(float %3)
  ret float %4
}

declare float @__nv_coshf(float) #3

define double @__hc_cosh_double(double) #0 {
  %2 = alloca double, align 8
  store double %0, double* %2, align 8
  %3 = load double, double* %2, align 8
  %4 = call double @__nv_cosh(double %3)
  ret double %4
}

declare double @__nv_cosh(double) #3

define float @__hc_cospi(float) #0 {
  %2 = alloca float, align 4
  store float %0, float* %2, align 4
  %3 = load float, float* %2, align 4
  %4 = call float @__nv_cospif(float %3)
  ret float %4
}

declare float @__nv_cospif(float) #3

define double @__hc_cospi_double(double) #0 {
  %2 = alloca double, align 8
  store double %0, double* %2, align 8
  %3 = load double, double* %2, align 8
  %4 = call double @__nv_cospi(double %3)
  ret double %4
}

declare double @__nv_cospi(double) #3

define float @__hc_erf(float) #0 {
  %2 = alloca float, align 4
  store float %0, float* %2, align 4
  %3 = load float, float* %2, align 4
  %4 = call float @__nv_erff(float %3)
  ret float %4
}

declare float @__nv_erff(float) #3

define double @__hc_erf_double(double) #0 {
  %2 = alloca double, align 8
  store double %0, double* %2, align 8
  %3 = load double, double* %2, align 8
  %4 = call double @__nv_erf(double %3)
  ret double %4
}

declare double @__nv_erf(double) #3

define float @__hc_erfc(float) #0 {
  %2 = alloca float, align 4
  store float %0, float* %2, align 4
  %3 = load float, float* %2, align 4
  %4 = call float @__nv_erfcf(float %3)
  ret float %4
}

declare float @__nv_erfcf(float) #3

define double @__hc_erfc_double(double) #0 {
  %2 = alloca double, align 8
  store double %0, double* %2, align 8
  %3 = load double, double* %2, align 8
  %4 = call double @__nv_erfc(double %3)
  ret double %4
}

declare double @__nv_erfc(double) #3

define float @__hc_exp(float) #0 {
  %2 = alloca float, align 4
  store float %0, float* %2, align 4
  %3 = load float, float* %2, align 4
  %4 = call float @__nv_fast_expf(float %3)
  ret float %4
}

declare float @__nv_fast_expf(float) #3

define double @__hc_exp_double(double) #0 {
  %2 = alloca double, align 8
  store double %0, double* %2, align 8
  %3 = load double, double* %2, align 8
  %4 = call double @__nv_exp(double %3)
  ret double %4
}

declare double @__nv_exp(double) #3

define float @__hc_exp2(float) #0 {
  %2 = alloca float, align 4
  store float %0, float* %2, align 4
  %3 = load float, float* %2, align 4
  %4 = call float @__nv_exp2f(float %3)
  ret float %4
}

declare float @__nv_exp2f(float) #3

define double @__hc_exp2_double(double) #0 {
  %2 = alloca double, align 8
  store double %0, double* %2, align 8
  %3 = load double, double* %2, align 8
  %4 = call double @__nv_exp2(double %3)
  ret double %4
}

declare double @__nv_exp2(double) #3

define float @__hc_exp10(float) #0 {
  %2 = alloca float, align 4
  store float %0, float* %2, align 4
  %3 = load float, float* %2, align 4
  %4 = call float @__nv_fast_exp10f(float %3)
  ret float %4
}

declare float @__nv_fast_exp10f(float) #3

define double @__hc_exp10_double(double) #0 {
  %2 = alloca double, align 8
  store double %0, double* %2, align 8
  %3 = load double, double* %2, align 8
  %4 = call double @__nv_exp10(double %3)
  ret double %4
}

declare double @__nv_exp10(double) #3

define float @__hc_expm1(float) #0 {
  %2 = alloca float, align 4
  store float %0, float* %2, align 4
  %3 = load float, float* %2, align 4
  %4 = call float @__nv_expm1f(float %3)
  ret float %4
}

declare float @__nv_expm1f(float) #3

define double @__hc_expm1_double(double) #0 {
  %2 = alloca double, align 8
  store double %0, double* %2, align 8
  %3 = load double, double* %2, align 8
  %4 = call double @__nv_expm1(double %3)
  ret double %4
}

declare double @__nv_expm1(double) #3

define float @__hc_fabs(float) #0 {
  %2 = alloca float, align 4
  store float %0, float* %2, align 4
  %3 = load float, float* %2, align 4
  %4 = call float @__nv_fabsf(float %3)
  ret float %4
}

declare float @__nv_fabsf(float) #3

define double @__hc_fabs_double(double) #0 {
  %2 = alloca double, align 8
  store double %0, double* %2, align 8
  %3 = load double, double* %2, align 8
  %4 = call double @__nv_fabs(double %3)
  ret double %4
}

declare double @__nv_fabs(double) #3

define float @__hc_fdim(float, float) #0 {
  %3 = alloca float, align 4
  %4 = alloca float, align 4
  store float %0, float* %3, align 4
  store float %1, float* %4, align 4
  %5 = load float, float* %3, align 4
  %6 = load float, float* %4, align 4
  %7 = call float @__nv_fdimf(float %5, float %6)
  ret float %7
}

declare float @__nv_fdimf(float, float) #3

define double @__hc_fdim_double(double, double) #0 {
  %3 = alloca double, align 8
  %4 = alloca double, align 8
  store double %0, double* %3, align 8
  store double %1, double* %4, align 8
  %5 = load double, double* %3, align 8
  %6 = load double, double* %4, align 8
  %7 = call double @__nv_fdim(double %5, double %6)
  ret double %7
}

declare double @__nv_fdim(double, double) #3

define float @__hc_floor(float) #0 {
  %2 = alloca float, align 4
  store float %0, float* %2, align 4
  %3 = load float, float* %2, align 4
  %4 = call float @__nv_floorf(float %3)
  ret float %4
}

declare float @__nv_floorf(float) #3

define double @__hc_floor_double(double) #0 {
  %2 = alloca double, align 8
  store double %0, double* %2, align 8
  %3 = load double, double* %2, align 8
  %4 = call double @__nv_floor(double %3)
  ret double %4
}

declare double @__nv_floor(double) #3

define float @__hc_fma(float, float, float) #0 {
  %4 = alloca float, align 4
  %5 = alloca float, align 4
  %6 = alloca float, align 4
  store float %0, float* %4, align 4
  store float %1, float* %5, align 4
  store float %2, float* %6, align 4
  %7 = load float, float* %4, align 4
  %8 = load float, float* %5, align 4
  %9 = load float, float* %6, align 4
  %10 = call float @__nv_fmaf(float %7, float %8, float %9)
  ret float %10
}

declare float @__nv_fmaf(float, float, float) #3

define double @__hc_fma_double(double, double, double) #0 {
  %4 = alloca double, align 8
  %5 = alloca double, align 8
  %6 = alloca double, align 8
  store double %0, double* %4, align 8
  store double %1, double* %5, align 8
  store double %2, double* %6, align 8
  %7 = load double, double* %4, align 8
  %8 = load double, double* %5, align 8
  %9 = load double, double* %6, align 8
  %10 = call double @__nv_fma(double %7, double %8, double %9)
  ret double %10
}

declare double @__nv_fma(double, double, double) #3

define float @__hc_fmax(float, float) #0 {
  %3 = alloca float, align 4
  %4 = alloca float, align 4
  store float %0, float* %3, align 4
  store float %1, float* %4, align 4
  %5 = load float, float* %3, align 4
  %6 = load float, float* %4, align 4
  %7 = call float @__nv_fmaxf(float %5, float %6)
  ret float %7
}

declare float @__nv_fmaxf(float, float) #3

define double @__hc_fmax_double(double, double) #0 {
  %3 = alloca double, align 8
  %4 = alloca double, align 8
  store double %0, double* %3, align 8
  store double %1, double* %4, align 8
  %5 = load double, double* %3, align 8
  %6 = load double, double* %4, align 8
  %7 = call double @__nv_fmax(double %5, double %6)
  ret double %7
}

declare double @__nv_fmax(double, double) #3

define float @__hc_fmin(float, float) #0 {
  %3 = alloca float, align 4
  %4 = alloca float, align 4
  store float %0, float* %3, align 4
  store float %1, float* %4, align 4
  %5 = load float, float* %3, align 4
  %6 = load float, float* %4, align 4
  %7 = call float @__nv_fminf(float %5, float %6)
  ret float %7
}

declare float @__nv_fminf(float, float) #3

define double @__hc_fmin_double(double, double) #0 {
  %3 = alloca double, align 8
  %4 = alloca double, align 8
  store double %0, double* %3, align 8
  store double %1, double* %4, align 8
  %5 = load double, double* %3, align 8
  %6 = load double, double* %4, align 8
  %7 = call double @__nv_fmin(double %5, double %6)
  ret double %7
}

declare double @__nv_fmin(double, double) #3

define float @__hc_fmod(float, float) #0 {
  %3 = alloca float, align 4
  %4 = alloca float, align 4
  store float %0, float* %3, align 4
  store float %1, float* %4, align 4
  %5 = load float, float* %3, align 4
  %6 = load float, float* %4, align 4
  %7 = call float @__nv_fmodf(float %5, float %6)
  ret float %7
}

declare float @__nv_fmodf(float, float) #3

define double @__hc_fmod_double(double, double) #0 {
  %3 = alloca double, align 8
  %4 = alloca double, align 8
  store double %0, double* %3, align 8
  store double %1, double* %4, align 8
  %5 = load double, double* %3, align 8
  %6 = load double, double* %4, align 8
  %7 = call double @__nv_fmod(double %5, double %6)
  ret double %7
}

declare double @__nv_fmod(double, double) #3

define float @__hc_frexpf(float, i32*) #0 {
  %3 = alloca float, align 4
  %4 = alloca i32*, align 8
  store float %0, float* %3, align 4
  store i32* %1, i32** %4, align 8
  %5 = load float, float* %3, align 4
  %6 = load i32*, i32** %4, align 8
  %7 = call float @__nv_frexpf(float %5, i32* %6)
  ret float %7
}

declare float @__nv_frexpf(float, i32*) #3

define float @__hc_frexpf_global(float, i32 addrspace(1)*) #0 {
  %3 = alloca float, align 4
  %4 = alloca i32 addrspace(1)*, align 8
  store float %0, float* %3, align 4
  store i32 addrspace(1)* %1, i32 addrspace(1)** %4, align 8
  %5 = load float, float* %3, align 4
  %6 = load i32 addrspace(1)*, i32 addrspace(1)** %4, align 8
  %7 = addrspacecast i32 addrspace(1)* %6 to i32*
  %8 = call float @__nv_frexpf(float %5, i32* %7)
  ret float %8
}

define float @__hc_frexpf_local(float, i32*) #0 {
  %3 = alloca float, align 4
  %4 = alloca i32 addrspace(3)*, align 8
  store float %0, float* %3, align 4
  %cast = addrspacecast i32* %1 to i32 addrspace(3)*
  store i32 addrspace(3)* %cast, i32 addrspace(3)** %4, align 8
  %5 = load float, float* %3, align 4
  %6 = load i32 addrspace(3)*, i32 addrspace(3)** %4, align 8
  %7 = addrspacecast i32 addrspace(3)* %6 to i32*
  %8 = call float @__nv_frexpf(float %5, i32* %7)
  ret float %8
}

define double @__hc_frexp(double, i32*) #0 {
  %3 = alloca double, align 8
  %4 = alloca i32*, align 8
  store double %0, double* %3, align 8
  store i32* %1, i32** %4, align 8
  %5 = load double, double* %3, align 8
  %6 = load i32*, i32** %4, align 8
  %7 = call double @__nv_frexp(double %5, i32* %6)
  ret double %7
}

declare double @__nv_frexp(double, i32*) #3

define double @__hc_frexp_global(double, i32 addrspace(1)*) #0 {
  %3 = alloca double, align 8
  %4 = alloca i32 addrspace(1)*, align 8
  store double %0, double* %3, align 8
  store i32 addrspace(1)* %1, i32 addrspace(1)** %4, align 8
  %5 = load double, double* %3, align 8
  %6 = load i32 addrspace(1)*, i32 addrspace(1)** %4, align 8
  %7 = addrspacecast i32 addrspace(1)* %6 to i32*
  %8 = call double @__nv_frexp(double %5, i32* %7)
  ret double %8
}

define double @__hc_frexp_local(double, i32*) #0 {
  %3 = alloca double, align 8
  %4 = alloca i32 addrspace(3)*, align 8
  store double %0, double* %3, align 8
  %cast = addrspacecast i32* %1 to i32 addrspace(3)*
  store i32 addrspace(3)* %cast, i32 addrspace(3)** %4, align 8
  %5 = load double, double* %3, align 8
  %6 = load i32 addrspace(3)*, i32 addrspace(3)** %4, align 8
  %7 = addrspacecast i32 addrspace(3)* %6 to i32*
  %8 = call double @__nv_frexp(double %5, i32* %7)
  ret double %8
}

define float @__hc_hypot(float, float) #0 {
  %3 = alloca float, align 4
  %4 = alloca float, align 4
  store float %0, float* %3, align 4
  store float %1, float* %4, align 4
  %5 = load float, float* %3, align 4
  %6 = load float, float* %4, align 4
  %7 = call float @__nv_hypotf(float %5, float %6)
  ret float %7
}

declare float @__nv_hypotf(float, float) #3

define double @__hc_hypot_double(double, double) #0 {
  %3 = alloca double, align 8
  %4 = alloca double, align 8
  store double %0, double* %3, align 8
  store double %1, double* %4, align 8
  %5 = load double, double* %3, align 8
  %6 = load double, double* %4, align 8
  %7 = call double @__nv_hypot(double %5, double %6)
  ret double %7
}

declare double @__nv_hypot(double, double) #3

define i32 @__hc_ilogb(float) #0 {
  %2 = alloca float, align 4
  store float %0, float* %2, align 4
  %3 = load float, float* %2, align 4
  %4 = call i32 @__nv_ilogbf(float %3)
  ret i32 %4
}

declare i32 @__nv_ilogbf(float) #3

define i32 @__hc_ilogb_double(double) #0 {
  %2 = alloca double, align 8
  store double %0, double* %2, align 8
  %3 = load double, double* %2, align 8
  %4 = call i32 @__nv_ilogb(double %3)
  ret i32 %4
}

declare i32 @__nv_ilogb(double) #3

define i32 @__hc_isfinite(float) #0 {
  %2 = alloca float, align 4
  store float %0, float* %2, align 4
  %3 = load float, float* %2, align 4
  %4 = call i32 @__nv_finitef(float %3)
  ret i32 %4
}

declare i32 @__nv_finitef(float) #3

define i32 @__hc_isfinite_double(double) #0 {
  %2 = alloca double, align 8
  store double %0, double* %2, align 8
  %3 = load double, double* %2, align 8
  %4 = call i32 @__nv_isfinited(double %3)
  ret i32 %4
}

declare i32 @__nv_isfinited(double) #3

define i32 @__hc_isinf(float) #0 {
  %2 = alloca float, align 4
  store float %0, float* %2, align 4
  %3 = load float, float* %2, align 4
  %4 = call i32 @__nv_isinff(float %3)
  ret i32 %4
}

declare i32 @__nv_isinff(float) #3

define i32 @__hc_isinf_double(double) #0 {
  %2 = alloca double, align 8
  store double %0, double* %2, align 8
  %3 = load double, double* %2, align 8
  %4 = call i32 @__nv_isinfd(double %3)
  ret i32 %4
}

declare i32 @__nv_isinfd(double) #3

define i32 @__hc_isnan(float) #0 {
  %2 = alloca float, align 4
  store float %0, float* %2, align 4
  %3 = load float, float* %2, align 4
  %4 = call i32 @__nv_isnanf(float %3)
  ret i32 %4
}

declare i32 @__nv_isnanf(float) #3

define i32 @__hc_isnan_double(double) #0 {
  %2 = alloca double, align 8
  store double %0, double* %2, align 8
  %3 = load double, double* %2, align 8
  %4 = call i32 @__nv_isnand(double %3)
  ret i32 %4
}

declare i32 @__nv_isnand(double) #3

; Function Attrs: nounwind
define i32 @__hc_isnormal(float) #4 {
  %2 = fcmp oeq float %0, 0.000000e+00
  br i1 %2, label %9, label %3

; <label>:3:                                      ; preds = %1
  %4 = tail call i32 @__nv_isinff(float %0)
  %5 = icmp eq i32 %4, 0
  br i1 %5, label %6, label %9

; <label>:6:                                      ; preds = %3
  %7 = tail call i32 @__nv_isnanf(float %0)
  %8 = icmp eq i32 %7, 0
  br label %9

; <label>:9:                                      ; preds = %3, %6, %1
  %10 = phi i1 [ false, %3 ], [ false, %1 ], [ %8, %6 ]
  %11 = zext i1 %10 to i32
  ret i32 %11
}

; Function Attrs: nounwind
define i32 @__hc_isnormal_double(double) #4 {
   %2 = fcmp oeq double %0, 0.000000e+00
  br i1 %2, label %9, label %3

; <label>:3:                                      ; preds = %1
  %4 = tail call i32 @__nv_isinfd(double %0)
  %5 = icmp eq i32 %4, 0
  br i1 %5, label %6, label %9

; <label>:6:                                      ; preds = %3
  %7 = tail call i32 @__nv_isnand(double %0)
  %8 = icmp eq i32 %7, 0
  br label %9

; <label>:9:                                      ; preds = %3, %6, %1
  %10 = phi i1 [ false, %3 ], [ false, %1 ], [ %8, %6 ]
  %11 = zext i1 %10 to i32
  ret i32 %11
}

define float @__hc_log(float) #0 {
  %2 = alloca float, align 4
  store float %0, float* %2, align 4
  %3 = load float, float* %2, align 4
  %4 = call float @__nv_fast_logf(float %3)
  ret float %4
}

declare float @__nv_fast_logf(float) #3

define double @__hc_log_double(double) #0 {
  %2 = alloca double, align 8
  store double %0, double* %2, align 8
  %3 = load double, double* %2, align 8
  %4 = call double @__nv_log(double %3)
  ret double %4
}

declare double @__nv_log(double) #3

define float @__hc_log10(float) #0 {
  %2 = alloca float, align 4
  store float %0, float* %2, align 4
  %3 = load float, float* %2, align 4
  %4 = call float @__nv_fast_log10f(float %3)
  ret float %4
}

declare float @__nv_fast_log10f(float) #3

define double @__hc_log10_double(double) #0 {
  %2 = alloca double, align 8
  store double %0, double* %2, align 8
  %3 = load double, double* %2, align 8
  %4 = call double @__nv_log10(double %3)
  ret double %4
}

declare double @__nv_log10(double) #3

define float @__hc_log2(float) #0 {
  %2 = alloca float, align 4
  store float %0, float* %2, align 4
  %3 = load float, float* %2, align 4
  %4 = call float @__nv_fast_log2f(float %3)
  ret float %4
}

declare float @__nv_fast_log2f(float) #3

define double @__hc_log2_double(double) #0 {
  %2 = alloca double, align 8
  store double %0, double* %2, align 8
  %3 = load double, double* %2, align 8
  %4 = call double @__nv_log2(double %3)
  ret double %4
}

declare double @__nv_log2(double) #3

define float @__hc_log1p(float) #0 {
  %2 = alloca float, align 4
  store float %0, float* %2, align 4
  %3 = load float, float* %2, align 4
  %4 = call float @__nv_log1pf(float %3)
  ret float %4
}

declare float @__nv_log1pf(float) #3

define double @__hc_log1p_double(double) #0 {
  %2 = alloca double, align 8
  store double %0, double* %2, align 8
  %3 = load double, double* %2, align 8
  %4 = call double @__nv_log1p(double %3)
  ret double %4
}

declare double @__nv_log1p(double) #3

define float @__hc_logb(float) #0 {
  %2 = alloca float, align 4
  store float %0, float* %2, align 4
  %3 = load float, float* %2, align 4
  %4 = call float @__nv_logbf(float %3)
  ret float %4
}

declare float @__nv_logbf(float) #3

define double @__hc_logb_double(double) #0 {
  %2 = alloca double, align 8
  store double %0, double* %2, align 8
  %3 = load double, double* %2, align 8
  %4 = call double @__nv_logb(double %3)
  ret double %4
}

declare double @__nv_logb(double) #3

define float @__hc_modff(float, float*) #0 {
  %3 = alloca float, align 4
  %4 = alloca float*, align 8
  store float %0, float* %3, align 4
  store float* %1, float** %4, align 8
  %5 = load float, float* %3, align 4
  %6 = load float*, float** %4, align 8
  %7 = call float @__nv_modff(float %5, float* %6)
  ret float %7
}

declare float @__nv_modff(float, float*) #3

define float @__hc_modff_global(float, float addrspace(1)*) #0 {
  %3 = alloca float, align 4
  %4 = alloca float addrspace(1)*, align 8
  store float %0, float* %3, align 4
  store float addrspace(1)* %1, float addrspace(1)** %4, align 8
  %5 = load float, float* %3, align 4
  %6 = load float addrspace(1)*, float addrspace(1)** %4, align 8
  %7 = addrspacecast float addrspace(1)* %6 to float*
  %8 = call float @__nv_modff(float %5, float* %7)
  ret float %8
}

define float @__hc_modff_local(float, float*) #0 {
  %3 = alloca float, align 4
  %4 = alloca float addrspace(3)*, align 8
  store float %0, float* %3, align 4
  %cast = addrspacecast float* %1 to float addrspace(3)*
  store float addrspace(3)* %cast, float addrspace(3)** %4, align 8
  %5 = load float, float* %3, align 4
  %6 = load float addrspace(3)*, float addrspace(3)** %4, align 8
  %7 = addrspacecast float addrspace(3)* %6 to float*
  %8 = call float @__nv_modff(float %5, float* %7)
  ret float %8
}

define double @__hc_modf(double, double*) #0 {
  %3 = alloca double, align 8
  %4 = alloca double*, align 8
  store double %0, double* %3, align 8
  store double* %1, double** %4, align 8
  %5 = load double, double* %3, align 8
  %6 = load double*, double** %4, align 8
  %7 = call double @__nv_modf(double %5, double* %6)
  ret double %7
}

declare double @__nv_modf(double, double*) #3

define double @__hc_modf_global(double, double addrspace(1)*) #0 {
  %3 = alloca double, align 8
  %4 = alloca double addrspace(1)*, align 8
  store double %0, double* %3, align 8
  store double addrspace(1)* %1, double addrspace(1)** %4, align 8
  %5 = load double, double* %3, align 8
  %6 = load double addrspace(1)*, double addrspace(1)** %4, align 8
  %7 = addrspacecast double addrspace(1)* %6 to double*
  %8 = call double @__nv_modf(double %5, double* %7)
  ret double %8
}

define double @__hc_modf_local(double, double*) #0 {
  %3 = alloca double, align 8
  %4 = alloca double addrspace(3)*, align 8
  store double %0, double* %3, align 8
  %cast = addrspacecast double* %1 to double addrspace(3)*
  store double addrspace(3)* %cast, double addrspace(3)** %4, align 8
  %5 = load double, double* %3, align 8
  %6 = load double addrspace(3)*, double addrspace(3)** %4, align 8
  %7 = addrspacecast double addrspace(3)* %6 to double*
  %8 = call double @__nv_modf(double %5, double* %7)
  ret double %8
}

define float @__hc_nan(i8*) #0 {
  %2 = alloca i8*, align 8
  store i8* %0, i8** %2, align 8
  %3 = load i8*, i8** %2, align 8
  %4 = call float @__nv_nanf(i8* %3)
  ret float %4
}

declare float @__nv_nanf(i8*) #3

define double @__hc_nan_double(i8*) #0 {
  %2 = alloca i8*, align 8
  store i8* %0, i8** %2, align 8
  %3 = load i8*, i8** %2, align 8
  %4 = call double @__nv_nan(i8* %3)
  ret double %4
}

declare double @__nv_nan(i8*) #3

define float @__hc_nearbyint(float) #0 {
  %2 = alloca float, align 4
  store float %0, float* %2, align 4
  %3 = load float, float* %2, align 4
  %4 = call float @__nv_nearbyintf(float %3)
  ret float %4
}

declare float @__nv_nearbyintf(float) #3

define double @__hc_nearbyint_double(double) #0 {
  %2 = alloca double, align 8
  store double %0, double* %2, align 8
  %3 = load double, double* %2, align 8
  %4 = call double @__nv_nearbyint(double %3)
  ret double %4
}

declare double @__nv_nearbyint(double) #3

define float @__hc_nextafter(float, float) #0 {
  %3 = alloca float, align 4
  %4 = alloca float, align 4
  store float %0, float* %3, align 4
  store float %1, float* %4, align 4
  %5 = load float, float* %3, align 4
  %6 = load float, float* %4, align 4
  %7 = call float @__nv_nextafterf(float %5, float %6)
  ret float %7
}

declare float @__nv_nextafterf(float, float) #3

define double @__hc_nextafter_double(double, double) #0 {
  %3 = alloca double, align 8
  %4 = alloca double, align 8
  store double %0, double* %3, align 8
  store double %1, double* %4, align 8
  %5 = load double, double* %3, align 8
  %6 = load double, double* %4, align 8
  %7 = call double @__nv_nextafter(double %5, double %6)
  ret double %7
}

declare double @__nv_nextafter(double, double) #3

define float @__hc_pow(float, float) #0 {
  %3 = alloca float, align 4
  %4 = alloca float, align 4
  store float %0, float* %3, align 4
  store float %1, float* %4, align 4
  %5 = load float, float* %3, align 4
  %6 = load float, float* %4, align 4
  %7 = call float @__nv_fast_powf(float %5, float %6)
  ret float %7
}

declare float @__nv_fast_powf(float, float) #3

define double @__hc_pow_double(double, double) #0 {
  %3 = alloca double, align 8
  %4 = alloca double, align 8
  store double %0, double* %3, align 8
  store double %1, double* %4, align 8
  %5 = load double, double* %3, align 8
  %6 = load double, double* %4, align 8
  %7 = call double @__nv_pow(double %5, double %6)
  ret double %7
}

declare double @__nv_pow(double, double) #3

define float @__hc_remainder(float, float) #0 {
  %3 = alloca float, align 4
  %4 = alloca float, align 4
  store float %0, float* %3, align 4
  store float %1, float* %4, align 4
  %5 = load float, float* %3, align 4
  %6 = load float, float* %4, align 4
  %7 = call float @__nv_remainderf(float %5, float %6)
  ret float %7
}

declare float @__nv_remainderf(float, float) #3

define double @__hc_remainder_double(double, double) #0 {
  %3 = alloca double, align 8
  %4 = alloca double, align 8
  store double %0, double* %3, align 8
  store double %1, double* %4, align 8
  %5 = load double, double* %3, align 8
  %6 = load double, double* %4, align 8
  %7 = call double @__nv_remainder(double %5, double %6)
  ret double %7
}

declare double @__nv_remainder(double, double) #3

define float @__hc_remquof(float, float, i32*) #0 {
  %4 = alloca float, align 4
  %5 = alloca float, align 4
  %6 = alloca i32*, align 8
  store float %0, float* %4, align 4
  store float %1, float* %5, align 4
  store i32* %2, i32** %6, align 8
  %7 = load float, float* %4, align 4
  %8 = load float, float* %5, align 4
  %9 = load i32*, i32** %6, align 8
  %10 = call float @__nv_remquof(float %7, float %8, i32* %9)
  ret float %10
}

declare float @__nv_remquof(float, float, i32*) #3

define float @__hc_remquof_global(float, float, i32 addrspace(1)*) #0 {
  %4 = alloca float, align 4
  %5 = alloca float, align 4
  %6 = alloca i32 addrspace(1)*, align 8
  store float %0, float* %4, align 4
  store float %1, float* %5, align 4
  store i32 addrspace(1)* %2, i32 addrspace(1)** %6, align 8
  %7 = load float, float* %4, align 4
  %8 = load float, float* %5, align 4
  %9 = load i32 addrspace(1)*, i32 addrspace(1)** %6, align 8
  %10 = addrspacecast i32 addrspace(1)* %9 to i32*
  %11 = call float @__nv_remquof(float %7, float %8, i32* %10)
  ret float %11
}

define float @__hc_remquof_local(float, float, i32 addrspace(3)*) #0 {
  %4 = alloca float, align 4
  %5 = alloca float, align 4
  %6 = alloca i32 addrspace(3)*, align 8
  store float %0, float* %4, align 4
  store float %1, float* %5, align 4
  store i32 addrspace(3)* %2, i32 addrspace(3)** %6, align 8
  %7 = load float, float* %4, align 4
  %8 = load float, float* %5, align 4
  %9 = load i32 addrspace(3)*, i32 addrspace(3)** %6, align 8
  %10 = addrspacecast i32 addrspace(3)* %9 to i32*
  %11 = call float @__nv_remquof(float %7, float %8, i32* %10)
  ret float %11
}

define double @__hc_remquo(double, double, i32*) #0 {
  %4 = alloca double, align 8
  %5 = alloca double, align 8
  %6 = alloca i32*, align 8
  store double %0, double* %4, align 8
  store double %1, double* %5, align 8
  store i32* %2, i32** %6, align 8
  %7 = load double, double* %4, align 8
  %8 = load double, double* %5, align 8
  %9 = load i32*, i32** %6, align 8
  %10 = call double @__nv_remquo(double %7, double %8, i32* %9)
  ret double %10
}

declare double @__nv_remquo(double, double, i32*) #3

define double @__hc_remquo_global(double, double, i32 addrspace(1)*) #0 {
  %4 = alloca double, align 8
  %5 = alloca double, align 8
  %6 = alloca i32 addrspace(1)*, align 8
  store double %0, double* %4, align 8
  store double %1, double* %5, align 8
  store i32 addrspace(1)* %2, i32 addrspace(1)** %6, align 8
  %7 = load double, double* %4, align 8
  %8 = load double, double* %5, align 8
  %9 = load i32 addrspace(1)*, i32 addrspace(1)** %6, align 8
  %10 = addrspacecast i32 addrspace(1)* %9 to i32*
  %11 = call double @__nv_remquo(double %7, double %8, i32* %10)
  ret double %11
}

define double @__hc_remquo_local(double, double, i32 addrspace(3)*) #0 {
  %4 = alloca double, align 8
  %5 = alloca double, align 8
  %6 = alloca i32 addrspace(3)*, align 8
  store double %0, double* %4, align 8
  store double %1, double* %5, align 8
  store i32 addrspace(3)* %2, i32 addrspace(3)** %6, align 8
  %7 = load double, double* %4, align 8
  %8 = load double, double* %5, align 8
  %9 = load i32 addrspace(3)*, i32 addrspace(3)** %6, align 8
  %10 = addrspacecast i32 addrspace(3)* %9 to i32*
  %11 = call double @__nv_remquo(double %7, double %8, i32* %10)
  ret double %11
}

define float @__hc_round(float) #0 {
  %2 = alloca float, align 4
  store float %0, float* %2, align 4
  %3 = load float, float* %2, align 4
  %4 = call float @__nv_roundf(float %3)
  ret float %4
}

declare float @__nv_roundf(float) #3

define double @__hc_round_double(double) #0 {
  %2 = alloca double, align 8
  store double %0, double* %2, align 8
  %3 = load double, double* %2, align 8
  %4 = call double @__nv_round(double %3)
  ret double %4
}

declare double @__nv_round(double) #3

define float @__hc_rsqrt(float) #0 {
  %2 = alloca float, align 4
  store float %0, float* %2, align 4
  %3 = load float, float* %2, align 4
  %4 = call float @__nv_rsqrtf(float %3)
  ret float %4
}

declare float @__nv_rsqrtf(float) #3

define double @__hc_rsqrt_double(double) #0 {
  %2 = alloca double, align 8
  store double %0, double* %2, align 8
  %3 = load double, double* %2, align 8
  %4 = call double @__nv_rsqrt(double %3)
  ret double %4
}

declare double @__nv_rsqrt(double) #3

define float @__hc_sinpi(float) #0 {
  %2 = alloca float, align 4
  store float %0, float* %2, align 4
  %3 = load float, float* %2, align 4
  %4 = call float @__nv_sinpif(float %3)
  ret float %4
}

declare float @__nv_sinpif(float) #3

define double @__hc_sinpi_double(double) #0 {
  %2 = alloca double, align 8
  store double %0, double* %2, align 8
  %3 = load double, double* %2, align 8
  %4 = call double @__nv_sinpi(double %3)
  ret double %4
}

declare double @__nv_sinpi(double) #3

define float @__hc_ldexp(float, i32) #0 {
  %3 = alloca float, align 4
  %4 = alloca i32, align 4
  store float %0, float* %3, align 4
  store i32 %1, i32* %4, align 4
  %5 = load float, float* %3, align 4
  %6 = load i32, i32* %4, align 4
  %7 = call float @__nv_ldexpf(float %5, i32 %6)
  ret float %7
}

declare float @__nv_ldexpf(float, i32) #3

define double @__hc_ldexp_double(double, i32) #0 {
  %3 = alloca double, align 8
  %4 = alloca i32, align 4
  store double %0, double* %3, align 8
  store i32 %1, i32* %4, align 4
  %5 = load double, double* %3, align 8
  %6 = load i32, i32* %4, align 4
  %7 = call double @__nv_ldexp(double %5, i32 %6)
  ret double %7
}

declare double @__nv_ldexp(double, i32) #3

define i32 @__hc_signbit(float) #0 {
  %2 = alloca float, align 4
  store float %0, float* %2, align 4
  %3 = load float, float* %2, align 4
  %4 = call i32 @__nv_signbitf(float %3)
  ret i32 %4
}

declare i32 @__nv_signbitf(float) #3

define i32 @__hc_signbit_double(double) #0 {
  %2 = alloca double, align 8
  store double %0, double* %2, align 8
  %3 = load double, double* %2, align 8
  %4 = call i32 @__nv_signbitd(double %3)
  ret i32 %4
}

declare i32 @__nv_signbitd(double) #3

define float @__hc_sin(float) #0 {
  %2 = alloca float, align 4
  store float %0, float* %2, align 4
  %3 = load float, float* %2, align 4
  %4 = call float @__nv_sinf(float %3)
  ret float %4
}

declare float @__nv_sinf(float) #3

define double @__hc_sin_double(double) #0 {
  %2 = alloca double, align 8
  store double %0, double* %2, align 8
  %3 = load double, double* %2, align 8
  %4 = call double @__nv_sin(double %3)
  ret double %4
}

declare double @__nv_sin(double) #3

define float @__hc_sinf(float) #0 {
  %2 = alloca float, align 4
  store float %0, float* %2, align 4
  %3 = load float, float* %2, align 4
  %4 = call float @__nv_sinf(float %3)
  ret float %4
}

define void @__hc_sincosf(float, float*, float*) #0 {
  %4 = alloca float, align 4
  %5 = alloca float*, align 8
  %6 = alloca float*, align 8
  store float %0, float* %4, align 4
  store float* %1, float** %5, align 8
  store float* %2, float** %6, align 8
  %7 = load float, float* %4, align 4
  %8 = load float*, float** %5, align 8
  %9 = load float*, float** %6, align 8
  call void @__nv_fast_sincosf(float %7, float* %8, float* %9)
  ret void
}

declare void @__nv_fast_sincosf(float, float*, float*) #3

define void @__hc_sincosf_global(float, float addrspace(1)*, float addrspace(1)*) #0 {
  %4 = alloca float, align 4
  %5 = alloca float addrspace(1)*, align 8
  %6 = alloca float addrspace(1)*, align 8
  store float %0, float* %4, align 4
  store float addrspace(1)* %1, float addrspace(1)** %5, align 8
  store float addrspace(1)* %2, float addrspace(1)** %6, align 8
  %7 = load float, float* %4, align 4
  %8 = load float addrspace(1)*, float addrspace(1)** %5, align 8
  %9 = addrspacecast float addrspace(1)* %8 to float*
  %10 = load float addrspace(1)*, float addrspace(1)** %6, align 8
  %11 = addrspacecast float addrspace(1)* %10 to float*
  call void @__nv_fast_sincosf(float %7, float* %9, float* %11)
  ret void
}

define void @__hc_sincosf_local(float, float addrspace(3)*, float addrspace(3)*) #0 {
  %4 = alloca float, align 4
  %5 = alloca float addrspace(3)*, align 8
  %6 = alloca float addrspace(3)*, align 8
  store float %0, float* %4, align 4
  store float addrspace(3)* %1, float addrspace(3)** %5, align 8
  store float addrspace(3)* %2, float addrspace(3)** %6, align 8
  %7 = load float, float* %4, align 4
  %8 = load float addrspace(3)*, float addrspace(3)** %5, align 8
  %9 = addrspacecast float addrspace(3)* %8 to float*
  %10 = load float addrspace(3)*, float addrspace(3)** %6, align 8
  %11 = addrspacecast float addrspace(3)* %10 to float*
  call void @__nv_fast_sincosf(float %7, float* %9, float* %11)
  ret void
}

define void @__hc_sincos_double(double, double*, double*) #0 {
  %4 = alloca double, align 8
  %5 = alloca double*, align 8
  %6 = alloca double*, align 8
  store double %0, double* %4, align 8
  store double* %1, double** %5, align 8
  store double* %2, double** %6, align 8
  %7 = load double, double* %4, align 8
  %8 = load double*, double** %5, align 8
  %9 = load double*, double** %6, align 8
  call void @__nv_sincos(double %7, double* %8, double* %9)
  ret void
}

declare void @__nv_sincos(double, double*, double*) #3

define void @__hc_sincos_double_global(double, double addrspace(1)*, double addrspace(1)*) #0 {
  %4 = alloca double, align 8
  %5 = alloca double addrspace(1)*, align 8
  %6 = alloca double addrspace(1)*, align 8
  store double %0, double* %4, align 8
  store double addrspace(1)* %1, double addrspace(1)** %5, align 8
  store double addrspace(1)* %2, double addrspace(1)** %6, align 8
  %7 = load double, double* %4, align 8
  %8 = load double addrspace(1)*, double addrspace(1)** %5, align 8
  %9 = addrspacecast double addrspace(1)* %8 to double*
  %10 = load double addrspace(1)*, double addrspace(1)** %6, align 8
  %11 = addrspacecast double addrspace(1)* %10 to double*
  call void @__nv_sincos(double %7, double* %9, double* %11)
  ret void
}

define void @__hc_sincos_double_local(double, double addrspace(3)*, double addrspace(3)*) #0 {
  %4 = alloca double, align 8
  %5 = alloca double addrspace(3)*, align 8
  %6 = alloca double addrspace(3)*, align 8
  store double %0, double* %4, align 8
  store double addrspace(3)* %1, double addrspace(3)** %5, align 8
  store double addrspace(3)* %2, double addrspace(3)** %6, align 8
  %7 = load double, double* %4, align 8
  %8 = load double addrspace(3)*, double addrspace(3)** %5, align 8
  %9 = addrspacecast double addrspace(3)* %8 to double*
  %10 = load double addrspace(3)*, double addrspace(3)** %6, align 8
  %11 = addrspacecast double addrspace(3)* %10 to double*
  call void @__nv_sincos(double %7, double* %9, double* %11)
  ret void
}

define float @__hc_sinh(float) #0 {
  %2 = alloca float, align 4
  store float %0, float* %2, align 4
  %3 = load float, float* %2, align 4
  %4 = call float @__nv_sinhf(float %3)
  ret float %4
}

declare float @__nv_sinhf(float) #3

define double @__hc_sinh_double(double) #0 {
  %2 = alloca double, align 8
  store double %0, double* %2, align 8
  %3 = load double, double* %2, align 8
  %4 = call double @__nv_sinh(double %3)
  ret double %4
}

declare double @__nv_sinh(double) #3

define float @__hc_sqrt(float) #0 {
  %2 = alloca float, align 4
  store float %0, float* %2, align 4
  %3 = load float, float* %2, align 4
  %4 = call float @__nv_sqrtf(float %3)
  ret float %4
}

declare float @__nv_sqrtf(float) #3

define double @__hc_sqrt_double(double) #0 {
  %2 = alloca double, align 8
  store double %0, double* %2, align 8
  %3 = load double, double* %2, align 8
  %4 = call double @__nv_sqrt(double %3)
  ret double %4
}

declare double @__nv_sqrt(double) #3

define float @__hc_tgamma(float) #0 {
  %2 = alloca float, align 4
  store float %0, float* %2, align 4
  %3 = load float, float* %2, align 4
  %4 = call float @__nv_tgammaf(float %3)
  ret float %4
}

declare float @__nv_tgammaf(float) #3

define double @__hc_tgamma_double(double) #0 {
  %2 = alloca double, align 8
  store double %0, double* %2, align 8
  %3 = load double, double* %2, align 8
  %4 = call double @__nv_tgamma(double %3)
  ret double %4
}

declare double @__nv_tgamma(double) #3

define float @__hc_tan(float) #0 {
  %2 = alloca float, align 4
  store float %0, float* %2, align 4
  %3 = load float, float* %2, align 4
  %4 = call float @__nv_tanf(float %3)
  ret float %4
}

declare float @__nv_tanf(float) #3

define double @__hc_tan_double(double) #0 {
  %2 = alloca double, align 8
  store double %0, double* %2, align 8
  %3 = load double, double* %2, align 8
  %4 = call double @__nv_tan(double %3)
  ret double %4
}

declare double @__nv_tan(double) #3

define float @__hc_tanf(float) #0 {
  %2 = alloca float, align 4
  store float %0, float* %2, align 4
  %3 = load float, float* %2, align 4
  %4 = call float @__nv_tanf(float %3)
  ret float %4
}

define float @__hc_tanh(float) #0 {
  %2 = alloca float, align 4
  store float %0, float* %2, align 4
  %3 = load float, float* %2, align 4
  %4 = call float @__nv_tanhf(float %3)
  ret float %4
}

declare float @__nv_tanhf(float) #3

define double @__hc_tanh_double(double) #0 {
  %2 = alloca double, align 8
  store double %0, double* %2, align 8
  %3 = load double, double* %2, align 8
  %4 = call double @__nv_tanh(double %3)
  ret double %4
}

declare double @__nv_tanh(double) #3

define float @__hc_tanpi(float) #0 {
  %2 = alloca float, align 4
  store float %0, float* %2, align 4
  %3 = load float, float* %2, align 4
  %4 = fpext float %3 to double
  %5 = fmul double %4, 0x400921FB54442D18
  %6 = fptrunc double %5 to float
  %7 = call float @__nv_tanf(float %6)
  ret float %7
}

define double @__hc_tanpi_double(double) #0 {
  %2 = alloca double, align 8
  store double %0, double* %2, align 8
  %3 = load double, double* %2, align 8
  %4 = fmul double %3, 0x400921FB54442D18
  %5 = call double @__nv_tan(double %4)
  ret double %5
}

define float @__hc_trunc(float) #0 {
  %2 = alloca float, align 4
  store float %0, float* %2, align 4
  %3 = load float, float* %2, align 4
  %4 = call float @__nv_truncf(float %3)
  ret float %4
}

declare float @__nv_truncf(float) #3

define double @__hc_trunc_double(double) #0 {
  %2 = alloca double, align 8
  store double %0, double* %2, align 8
  %3 = load double, double* %2, align 8
  %4 = call double @__nv_trunc(double %3)
  ret double %4
}

declare double @__nv_trunc(double) #3

; Function Attrs: nounwind
define i32 @atomic_exchange_unsigned_global(i32 addrspace(1)*, i32) #0 {
  %ret = atomicrmw xchg i32 addrspace(1)* %0, i32 %1 seq_cst
  ret i32 %ret
}

; Function Attrs: nounwind
define i32 @atomic_exchange_unsigned_local(i32*, i32) #0 {
  %cast = addrspacecast i32* %0 to i32 addrspace(3)*
  %ret = atomicrmw xchg i32 addrspace(3)* %cast, i32 %1 seq_cst
  ret i32 %ret
}

; Function Attrs: nounwind
define i32 @atomic_exchange_unsigned(i32*, i32) #0 {
  %ret = atomicrmw xchg i32* %0, i32 %1 seq_cst
  ret i32 %ret
}

; Function Attrs: nounwind
define i32 @atomic_compare_exchange_unsigned_global(i32 addrspace(1)*, i32, i32) #0 {
  %val_success = cmpxchg i32 addrspace(1)* %0, i32 %1, i32 %2 seq_cst seq_cst
  %value_loaded = extractvalue { i32, i1 } %val_success, 0
  ret i32 %value_loaded
}

; Function Attrs: nounwind
define i32 @atomic_compare_exchange_unsigned_local(i32*, i32, i32) #0 {
  %cast = addrspacecast i32* %0 to i32 addrspace(3)*
  %val_success = cmpxchg i32 addrspace(3)* %cast, i32 %1, i32 %2 seq_cst seq_cst
  %value_loaded = extractvalue { i32, i1 } %val_success, 0
  ret i32 %value_loaded
}

; Function Attrs: nounwind
define i32 @atomic_compare_exchange_unsigned(i32* %x, i32 %y, i32 %z) #0 {
  %val_success = cmpxchg i32* %x, i32 %y, i32 %z seq_cst seq_cst
  %value_loaded = extractvalue { i32, i1 } %val_success, 0
  ret i32 %value_loaded
}

; Function Attrs: nounwind
define i32 @atomic_add_unsigned_global(i32 addrspace(1)*, i32) #0 {
  %ret = atomicrmw add i32 addrspace(1)* %0, i32 %1 seq_cst
  ret i32 %ret
}

; Function Attrs: nounwind
define i32 @atomic_add_unsigned_local(i32*, i32) #0 {
  %cast = addrspacecast i32* %0 to i32 addrspace(3)*
  %ret = atomicrmw add i32 addrspace(3)* %cast, i32 %1 seq_cst
  ret i32 %ret
}

; Function Attrs: nounwind
define i32 @atomic_add_unsigned(i32*, i32) #0 {
  %ret = atomicrmw add i32* %0, i32 %1 seq_cst
  ret i32 %ret
}

; Function Attrs: nounwind
define i32 @atomic_sub_unsigned_global(i32 addrspace(1)*, i32) #0 {
  %ret = atomicrmw sub i32 addrspace(1)* %0, i32 %1 seq_cst
  ret i32 %ret
}

; Function Attrs: nounwind
define i32 @atomic_sub_unsigned_local(i32*, i32) #0 {
  %cast = addrspacecast i32* %0 to i32 addrspace(3)*
  %ret = atomicrmw sub i32 addrspace(3)* %cast, i32 %1 seq_cst
  ret i32 %ret
}

; Function Attrs: nounwind
define i32 @atomic_sub_unsigned(i32*, i32) #0 {
  %ret = atomicrmw sub i32* %0, i32 %1 seq_cst
  ret i32 %ret
}

; Function Attrs: nounwind
define i32 @atomic_exchange_int_global(i32 addrspace(1)*, i32) #0 {
  %ret = atomicrmw xchg i32 addrspace(1)* %0, i32 %1 seq_cst
  ret i32 %ret
}

; Function Attrs: nounwind
define i32 @atomic_exchange_int_local(i32*, i32) #0 {
  %cast = addrspacecast i32* %0 to i32 addrspace(3)*
  %ret = atomicrmw xchg i32 addrspace(3)* %cast, i32 %1 seq_cst
  ret i32 %ret
}

; Function Attrs: nounwind
define i32 @atomic_exchange_int(i32*, i32) #0 {
  %ret = atomicrmw xchg i32* %0, i32 %1 seq_cst
  ret i32 %ret
}

; Function Attrs: nounwind
define i32 @atomic_compare_exchange_int_global(i32 addrspace(1)*, i32, i32) #0 {
  %val_success = cmpxchg i32 addrspace(1)* %0, i32 %1, i32 %2 seq_cst seq_cst
  %value_loaded = extractvalue { i32, i1 } %val_success, 0
  ret i32 %value_loaded
}

; Function Attrs: nounwind
define i32 @atomic_compare_exchange_int_local(i32*, i32, i32) #0 {
  %cast = addrspacecast i32* %0 to i32 addrspace(3)*
  %val_success = cmpxchg i32 addrspace(3)* %cast, i32 %1, i32 %2 seq_cst seq_cst
  %value_loaded = extractvalue { i32, i1 } %val_success, 0
  ret i32 %value_loaded
}

; Function Attrs: nounwind
define i32 @atomic_compare_exchange_int(i32*, i32, i32) #0 {
  %val_success = cmpxchg i32* %0, i32 %1, i32 %2 seq_cst seq_cst
  %value_loaded = extractvalue { i32, i1 } %val_success, 0
  ret i32 %value_loaded
}

; Function Attrs: nounwind
define i32 @atomic_add_int_global(i32 addrspace(1)*, i32) #0 {
  %ret = atomicrmw add i32 addrspace(1)* %0, i32 %1 seq_cst
  ret i32 %ret
}

; Function Attrs: nounwind
define i32 @atomic_add_int_local(i32*, i32) #0 {
  %cast = addrspacecast i32* %0 to i32 addrspace(3)*
  %ret = atomicrmw add i32 addrspace(3)* %cast, i32 %1 seq_cst
  ret i32 %ret
}

; Function Attrs: nounwind
define i32 @atomic_add_int(i32*, i32) #0 {
  %ret = atomicrmw add i32* %0, i32 %1 seq_cst
  ret i32 %ret
}

; Function Attrs: nounwind
define i32 @atomic_sub_int_global(i32 addrspace(1)*, i32) #0 {
  %ret = atomicrmw sub i32 addrspace(1)* %0, i32 %1 seq_cst
  ret i32 %ret
}

; Function Attrs: nounwind
define i32 @atomic_sub_int_local(i32*, i32) #0 {
  %cast = addrspacecast i32* %0 to i32 addrspace(3)*
  %ret = atomicrmw sub i32 addrspace(3)* %cast, i32 %1 seq_cst
  ret i32 %ret
}

; Function Attrs: nounwind
define i32 @atomic_sub_int(i32*, i32) #0 {
  %ret = atomicrmw sub i32* %0, i32 %1 seq_cst
  ret i32 %ret
}

; Function Attrs: nounwind
define float @atomic_exchange_float_global(float addrspace(1)*, float) #0 {
  %3 = bitcast float addrspace(1)* %0 to i32 addrspace(1)*
  %4 = bitcast float %1 to i32
  %5 = atomicrmw xchg i32 addrspace(1)* %3, i32 %4 seq_cst
  %6 = bitcast i32 %5 to float
  ret float %6
}

; Function Attrs: nounwind
define float @atomic_exchange_float_local(float*, float) #0 {
  %cast = addrspacecast float* %0 to float addrspace(3)*
  %3 = bitcast float addrspace(3)* %cast to i32 addrspace(3)*
  %4 = bitcast float %1 to i32
  %5 = atomicrmw xchg i32 addrspace(3)* %3, i32 %4 seq_cst
  %6 = bitcast i32 %5 to float
  ret float %6
}

; Function Attrs: nounwind
define float @atomic_exchange_float(float*, float) #0 {
  %3 = bitcast float* %0 to i32*
  %4 = bitcast float %1 to i32
  %5 = atomicrmw xchg i32* %3, i32 %4 seq_cst
  %6 = bitcast i32 %5 to float
  ret float %6
}

; Function Attrs: nounwind
define float @atomic_add_float_global(float addrspace(1)*, float) #0 {
  %3 = tail call float @llvm.nvvm.atomic.load.add.f32.p1f32(float addrspace(1)* %0, float %1) #6
  ret float %3
}

; Function Attrs: argmemonly nounwind
declare float @llvm.nvvm.atomic.load.add.f32.p1f32(float addrspace(1)* nocapture, float) #4

; Function Attrs: nounwind
define float @atomic_add_float_local(float*, float) #0 {
  %cast = addrspacecast float* %0 to float addrspace(3)*
  %3 = tail call float @llvm.nvvm.atomic.load.add.f32.p3f32(float addrspace(3)* %cast, float %1) #6
  ret float %3
}

; Function Attrs: argmemonly nounwind
declare float @llvm.nvvm.atomic.load.add.f32.p3f32(float addrspace(3)* nocapture, float) #4

; Function Attrs: nounwind
define float @atomic_add_float(float*, float) #0 {
  %3 = tail call float @llvm.nvvm.atomic.load.add.f32.p0f32(float* %0, float %1) #6
  ret float %3
}
; Function Attrs: argmemonly nounwind
declare float @llvm.nvvm.atomic.load.add.f32.p0f32(float* nocapture, float) #4

; Function Attrs: nounwind
define float @atomic_sub_float_global(float addrspace(1)*, float) #0 {
  %3 = fsub float 0.000000e+00, %1
  %4 = tail call float @llvm.nvvm.atomic.load.add.f32.p1f32(float addrspace(1)* %0, float %3) #6
  ret float %4
}

; Function Attrs: nounwind
define float @atomic_sub_float_local(float*, float) #0 {
  %cast = addrspacecast float* %0 to float addrspace(3)*
  %3 = fsub float 0.000000e+00, %1
  %4 = tail call float @llvm.nvvm.atomic.load.add.f32.p3f32(float addrspace(3)* %cast, float %3) #6
  ret float %4
}

; Function Attrs: nounwind
define float @atomic_sub_float(float*, float) #0 {
  %3 = fsub float 0.000000e+00, %1
  %4 = tail call float @llvm.nvvm.atomic.load.add.f32.p0f32(float* %0, float %3) #6
  ret float %4
}

; Function Attrs: nounwind
define i64 @atomic_exchange_uint64_global(i64 addrspace(1)*, i64) #0 {
  %ret = atomicrmw xchg i64 addrspace(1)* %0, i64 %1 seq_cst
  ret i64 %ret
}

; Function Attrs: nounwind
define i64 @atomic_exchange_uint64_local(i64*, i64) #0 {
  %cast = addrspacecast i64* %0 to i64 addrspace(3)*
  %ret = atomicrmw xchg i64 addrspace(3)* %cast, i64 %1 seq_cst
  ret i64 %ret
}

; Function Attrs: nounwind
define i64 @atomic_exchange_uint64(i64*, i64) #0 {
  %ret = atomicrmw xchg i64* %0, i64 %1 seq_cst
  ret i64 %ret
}

; Function Attrs: nounwind
define i64 @atomic_compare_exchange_uint64_global(i64 addrspace(1)*, i64, i64) #0 {
  %val_success = cmpxchg i64 addrspace(1)* %0, i64 %1, i64 %2 seq_cst seq_cst
  %value_loaded = extractvalue { i64, i1 } %val_success, 0
  ret i64 %value_loaded
}

; Function Attrs: nounwind
define i64 @atomic_compare_exchange_uint64_local(i64*, i64, i64) #0 {
  %cast = addrspacecast i64* %0 to i64 addrspace(3)*
  %val_success = cmpxchg i64 addrspace(3)* %cast, i64 %1, i64 %2 seq_cst seq_cst
  %value_loaded = extractvalue { i64, i1 } %val_success, 0
  ret i64 %value_loaded
}

; Function Attrs: nounwind
define i64 @atomic_compare_exchange_uint64(i64*, i64, i64) #0 {
  %val_success = cmpxchg i64* %0, i64 %1, i64 %2 seq_cst seq_cst
  %value_loaded = extractvalue { i64, i1 } %val_success, 0
  ret i64 %value_loaded
}

; Function Attrs: nounwind
define i64 @atomic_add_uint64_global(i64*, i64) #0 {
  %cast = addrspacecast i64* %0 to i64 addrspace(1)*
  %ret = atomicrmw add i64 addrspace(1)* %cast, i64 %1 seq_cst
  ret i64 %ret
}

; Function Attrs: nounwind
define i64 @atomic_add_uint64_local(i64*, i64) #0 {
  %cast = addrspacecast i64* %0 to i64 addrspace(3)*
  %ret = atomicrmw add i64 addrspace(3)* %cast, i64 %1 seq_cst
  ret i64 %ret
}

; Function Attrs: nounwind
define i64 @atomic_add_uint64(i64*, i64) #0 {
  %ret = atomicrmw add i64* %0, i64 %1 seq_cst
  ret i64 %ret
}

; Function Attrs: nounwind
define i32 @atomic_and_unsigned_global(i32 addrspace(1)*, i32) #0 {
  %ret = atomicrmw and i32 addrspace(1)* %0, i32 %1 seq_cst
  ret i32 %ret
}

define i32 @atomic_or_unsigned_global(i32 addrspace(1)*, i32) #0 {
  %ret = atomicrmw or i32 addrspace(1)* %0, i32 %1 seq_cst
  ret i32 %ret
}

define i32 @atomic_xor_unsigned_global(i32 addrspace(1)*, i32) #0 {
  %ret = atomicrmw xor i32 addrspace(1)* %0, i32 %1 seq_cst
  ret i32 %ret
}

define i32 @atomic_max_unsigned_global(i32 addrspace(1)*, i32) #0 {
  %ret = atomicrmw max i32 addrspace(1)* %0, i32 %1 seq_cst
  ret i32 %ret
}

define i32 @atomic_min_unsigned_global(i32 addrspace(1)*, i32) #0 {
  %ret = atomicrmw min i32 addrspace(1)* %0, i32 %1 seq_cst
  ret i32 %ret
}

; Function Attrs: nounwind
define i32 @atomic_and_unsigned_local(i32*, i32) #0 {
  %cast = addrspacecast i32* %0 to i32 addrspace(3)*
  %ret = atomicrmw and i32 addrspace(3)* %cast, i32 %1 seq_cst
  ret i32 %ret
}

; Function Attrs: nounwind
define i32 @atomic_or_unsigned_local(i32*, i32) #0 {
  %cast = addrspacecast i32* %0 to i32 addrspace(3)*
  %ret = atomicrmw or i32 addrspace(3)* %cast, i32 %1 seq_cst
  ret i32 %ret
}

; Function Attrs: nounwind
define i32 @atomic_xor_unsigned_local(i32*, i32) #0 {
  %cast = addrspacecast i32* %0 to i32 addrspace(3)*
  %ret = atomicrmw xor i32 addrspace(3)* %cast, i32 %1 seq_cst
  ret i32 %ret
}

; Function Attrs: nounwind
define i32 @atomic_max_unsigned_local(i32*, i32) #0 {
  %cast = addrspacecast i32* %0 to i32 addrspace(3)*
  %ret = atomicrmw max i32 addrspace(3)* %cast, i32 %1 seq_cst
  ret i32 %ret
}

; Function Attrs: nounwind
define i32 @atomic_min_unsigned_local(i32*, i32) #0 {
  %cast = addrspacecast i32* %0 to i32 addrspace(3)*
  %ret = atomicrmw min i32 addrspace(3)* %cast, i32 %1 seq_cst
  ret i32 %ret
}

; Function Attrs: nounwind
define i32 @atomic_and_unsigned(i32*, i32) #0 {
  %ret = atomicrmw and i32* %0, i32 %1 seq_cst
  ret i32 %ret
}

; Function Attrs: nounwind
define i32 @atomic_or_unsigned(i32*, i32) #0 {
  %ret = atomicrmw or i32* %0, i32 %1 seq_cst
  ret i32 %ret
}

; Function Attrs: nounwind
define i32 @atomic_xor_unsigned(i32*, i32) #0 {
  %ret = atomicrmw xor i32* %0, i32 %1 seq_cst
  ret i32 %ret
}

; Function Attrs: nounwind
define i32 @atomic_max_unsigned(i32*, i32) #0 {
  %ret = atomicrmw max i32* %0, i32 %1 seq_cst
  ret i32 %ret
}

; Function Attrs: nounwind
define i32 @atomic_min_unsigned(i32*, i32) #0 {
  %ret = atomicrmw min i32* %0, i32 %1 seq_cst
  ret i32 %ret
}

; Function Attrs: nounwind
define i32 @atomic_and_int_global(i32 addrspace(1)*, i32) #0 {
  %ret = atomicrmw and i32 addrspace(1)* %0, i32 %1 seq_cst
  ret i32 %ret
}

; Function Attrs: nounwind
define i32 @atomic_or_int_global(i32 addrspace(1)*, i32) #0 {
  %ret = atomicrmw or i32 addrspace(1)* %0, i32 %1 seq_cst
  ret i32 %ret
}

; Function Attrs: nounwind
define i32 @atomic_xor_int_global(i32 addrspace(1)*, i32) #0 {
  %ret = atomicrmw xor i32 addrspace(1)* %0, i32 %1 seq_cst
  ret i32 %ret
}

; Function Attrs: nounwind
define i32 @atomic_max_int_global(i32 addrspace(1)*, i32) #0 {
  %ret = atomicrmw max i32 addrspace(1)* %0, i32 %1 seq_cst
  ret i32 %ret
}

; Function Attrs: nounwind
define i32 @atomic_min_int_global(i32 addrspace(1)*, i32) #0 {
  %ret = atomicrmw min i32 addrspace(1)* %0, i32 %1 seq_cst
  ret i32 %ret
}

; Function Attrs: nounwind
define i32 @atomic_and_int_local(i32*, i32) #0 {
  %cast = addrspacecast i32* %0 to i32 addrspace(3)*
  %ret = atomicrmw and i32 addrspace(3)* %cast, i32 %1 seq_cst
  ret i32 %ret
}

; Function Attrs: nounwind
define i32 @atomic_or_int_local(i32*, i32) #0 {
  %cast = addrspacecast i32* %0 to i32 addrspace(3)*
  %ret = atomicrmw or i32 addrspace(3)* %cast, i32 %1 seq_cst
  ret i32 %ret
}

; Function Attrs: nounwind
define i32 @atomic_xor_int_local(i32*, i32) #0 {
  %cast = addrspacecast i32* %0 to i32 addrspace(3)*
  %ret = atomicrmw xor i32 addrspace(3)* %cast, i32 %1 seq_cst
  ret i32 %ret
}

; Function Attrs: nounwind
define i32 @atomic_max_int_local(i32*, i32) #0 {
  %cast = addrspacecast i32* %0 to i32 addrspace(3)*
  %ret = atomicrmw max i32 addrspace(3)* %cast, i32 %1 seq_cst
  ret i32 %ret
}

; Function Attrs: nounwind
define i32 @atomic_min_int_local(i32*, i32) #0 {
  %cast = addrspacecast i32* %0 to i32 addrspace(3)*
  %ret = atomicrmw min i32 addrspace(3)* %cast, i32 %1 seq_cst
  ret i32 %ret
}

; Function Attrs: nounwind
define i32 @atomic_and_int(i32*, i32) #0 {
  %ret = atomicrmw and i32* %0, i32 %1 seq_cst
  ret i32 %ret
}

; Function Attrs: nounwind
define i32 @atomic_or_int(i32*, i32) #0 {
  %ret = atomicrmw or i32* %0, i32 %1 seq_cst
  ret i32 %ret
}

; Function Attrs: nounwind
define i32 @atomic_max_int(i32*, i32) #0 {
  %ret = atomicrmw max i32* %0, i32 %1 seq_cst
  ret i32 %ret
}

; Function Attrs: nounwind
define i32 @atomic_min_int(i32*, i32) #0 {
  %ret = atomicrmw min i32* %0, i32 %1 seq_cst
  ret i32 %ret
}

; Function Attrs: nounwind
define i64 @atomic_and_uint64_global(i64 addrspace(1)*, i64) #0 {
  %3 = load i64, i64 addrspace(1)* %0, align 8
  br label %4

; <label>:4:                                      ; preds = %4, %2
  %5 = phi i64 [ %3, %2 ], [ %8, %4 ]
  %6 = and i64 %5, %1
  %7 = cmpxchg i64 addrspace(1)* %0, i64 %5, i64 %6 seq_cst seq_cst
  %8 = extractvalue { i64, i1 } %7, 0
  %9 = icmp eq i64 %5, %8
  br i1 %9, label %10, label %4

; <label>:10:                                     ; preds = %4
  %11 = phi i64 [ %5, %4 ]
  ret i64 %11
}

; Function Attrs: nounwind
define i64 @atomic_or_uint64_global(i64 addrspace(1)*, i64) #0 {
  %3 = load i64, i64 addrspace(1)* %0, align 8
  br label %4

; <label>:4:                                      ; preds = %4, %2
  %5 = phi i64 [ %3, %2 ], [ %8, %4 ]
  %6 = or i64 %5, %1
  %7 = cmpxchg i64 addrspace(1)* %0, i64 %5, i64 %6 seq_cst seq_cst
  %8 = extractvalue { i64, i1 } %7, 0
  %9 = icmp eq i64 %5, %8
  br i1 %9, label %10, label %4

; <label>:10:                                     ; preds = %4
  %11 = phi i64 [ %5, %4 ]
  ret i64 %11
}

; Function Attrs: nounwind
define i64 @atomic_xor_uint64_global(i64 addrspace(1)*, i64) #0 {
  %3 = load i64, i64 addrspace(1)* %0, align 8
  br label %4

; <label>:4:                                      ; preds = %4, %2
  %5 = phi i64 [ %3, %2 ], [ %8, %4 ]
  %6 = xor i64 %5, %1
  %7 = cmpxchg i64 addrspace(1)* %0, i64 %5, i64 %6 seq_cst seq_cst
  %8 = extractvalue { i64, i1 } %7, 0
  %9 = icmp eq i64 %5, %8
  br i1 %9, label %10, label %4

; <label>:10:                                     ; preds = %4
  %11 = phi i64 [ %5, %4 ]
  ret i64 %11
}

; Function Attrs: nounwind
define i64 @atomic_max_uint64_global(i64 addrspace(1)*, i64) #0 {
  %3 = load i64, i64 addrspace(1)* %0, align 8
  br label %4

; <label>:4:                                      ; preds = %4, %2
  %5 = phi i64 [ %3, %2 ], [ %9, %4 ]
  %6 = icmp ugt i64 %5, %1
  %7 = select i1 %6, i64 %5, i64 %1
  %8 = cmpxchg i64 addrspace(1)* %0, i64 %5, i64 %7 seq_cst seq_cst
  %9 = extractvalue { i64, i1 } %8, 0
  %10 = icmp eq i64 %5, %9
  br i1 %10, label %11, label %4

; <label>:11:                                     ; preds = %4
  %12 = phi i64 [ %5, %4 ]
  ret i64 %12
}

; Function Attrs: nounwind
define i64 @atomic_min_uint64_global(i64 addrspace(1)*, i64) #0 {
  %3 = load i64, i64 addrspace(1)* %0, align 8
  br label %4

; <label>:4:                                      ; preds = %4, %2
  %5 = phi i64 [ %3, %2 ], [ %9, %4 ]
  %6 = icmp ult i64 %5, %1
  %7 = select i1 %6, i64 %5, i64 %1
  %8 = cmpxchg i64 addrspace(1)* %0, i64 %5, i64 %7 seq_cst seq_cst
  %9 = extractvalue { i64, i1 } %8, 0
  %10 = icmp eq i64 %5, %9
  br i1 %10, label %11, label %4

; <label>:11:                                     ; preds = %4
  %12 = phi i64 [ %5, %4 ]
  ret i64 %12
}

; Function Attrs: nounwind
define i64 @atomic_and_uint64_local(i64*, i64) #0 {
  %cast = addrspacecast i64* %0 to i64 addrspace(3)*
  %3 = load i64, i64 addrspace(3)* %cast, align 8
  br label %4

; <label>:4:                                      ; preds = %4, %2
  %5 = phi i64 [ %3, %2 ], [ %8, %4 ]
  %6 = and i64 %5, %1
  %7 = cmpxchg i64 addrspace(3)* %cast, i64 %5, i64 %6 seq_cst seq_cst
  %8 = extractvalue { i64, i1 } %7, 0
  %9 = icmp eq i64 %5, %8
  br i1 %9, label %10, label %4

; <label>:10:                                     ; preds = %4
  %11 = phi i64 [ %5, %4 ]
  ret i64 %11
}

; Function Attrs: nounwind
define i64 @atomic_or_uint64_local(i64*, i64) #0 {
  %cast = addrspacecast i64* %0 to i64 addrspace(3)*
  %3 = load i64, i64 addrspace(3)* %cast, align 8
  br label %4

; <label>:4:                                      ; preds = %4, %2
  %5 = phi i64 [ %3, %2 ], [ %8, %4 ]
  %6 = or i64 %5, %1
  %7 = cmpxchg i64 addrspace(3)* %cast, i64 %5, i64 %6 seq_cst seq_cst
  %8 = extractvalue { i64, i1 } %7, 0
  %9 = icmp eq i64 %5, %8
  br i1 %9, label %10, label %4

; <label>:10:                                     ; preds = %4
  %11 = phi i64 [ %5, %4 ]
  ret i64 %11
}

; Function Attrs: nounwind
define i64 @atomic_xor_uint64_local(i64*, i64) #0 {
  %cast = addrspacecast i64* %0 to i64 addrspace(3)*
  %3 = load i64, i64 addrspace(3)* %cast, align 8
  br label %4

; <label>:4:                                      ; preds = %4, %2
  %5 = phi i64 [ %3, %2 ], [ %8, %4 ]
  %6 = xor i64 %5, %1
  %7 = cmpxchg i64 addrspace(3)* %cast, i64 %5, i64 %6 seq_cst seq_cst
  %8 = extractvalue { i64, i1 } %7, 0
  %9 = icmp eq i64 %5, %8
  br i1 %9, label %10, label %4

; <label>:10:                                     ; preds = %4
  %11 = phi i64 [ %5, %4 ]
  ret i64 %11
}

; Function Attrs: nounwind
define i64 @atomic_max_uint64_local(i64*, i64) #0 {
  %cast = addrspacecast i64* %0 to i64 addrspace(3)*
  %3 = load i64, i64 addrspace(3)* %cast, align 8
  br label %4

; <label>:4:                                      ; preds = %4, %2
  %5 = phi i64 [ %3, %2 ], [ %9, %4 ]
  %6 = icmp ugt i64 %5, %1
  %7 = select i1 %6, i64 %5, i64 %1
  %8 = cmpxchg i64 addrspace(3)* %cast, i64 %5, i64 %7 seq_cst seq_cst
  %9 = extractvalue { i64, i1 } %8, 0
  %10 = icmp eq i64 %5, %9
  br i1 %10, label %11, label %4

; <label>:11:                                     ; preds = %4
  %12 = phi i64 [ %5, %4 ]
  ret i64 %12
}

; Function Attrs: nounwind
define i64 @atomic_min_uint64_local(i64*, i64) #0 {
  %cast = addrspacecast i64* %0 to i64 addrspace(3)*
  %3 = load i64, i64 addrspace(3)* %cast, align 8
  br label %4

; <label>:4:                                      ; preds = %4, %2
  %5 = phi i64 [ %3, %2 ], [ %9, %4 ]
  %6 = icmp ult i64 %5, %1
  %7 = select i1 %6, i64 %5, i64 %1
  %8 = cmpxchg i64 addrspace(3)* %cast, i64 %5, i64 %7 seq_cst seq_cst
  %9 = extractvalue { i64, i1 } %8, 0
  %10 = icmp eq i64 %5, %9
  br i1 %10, label %11, label %4

; <label>:11:                                     ; preds = %4
  %12 = phi i64 [ %5, %4 ]
  ret i64 %12
}

; Function Attrs: nounwind
define i64 @atomic_and_uint64(i64*, i64) #0 {
  %3 = load i64, i64* %0, align 8
  br label %4

; <label>:4:                                      ; preds = %4, %2
  %5 = phi i64 [ %3, %2 ], [ %8, %4 ]
  %6 = and i64 %5, %1
  %7 = cmpxchg i64* %0, i64 %5, i64 %6 seq_cst seq_cst
  %8 = extractvalue { i64, i1 } %7, 0
  %9 = icmp eq i64 %5, %8
  br i1 %9, label %10, label %4

; <label>:10:                                     ; preds = %4
  %11 = phi i64 [ %5, %4 ]
  ret i64 %11
}

; Function Attrs: nounwind
define i64 @atomic_or_uint64(i64*, i64) #0 {
  %3 = load i64, i64* %0, align 8
  br label %4

; <label>:4:                                      ; preds = %4, %2
  %5 = phi i64 [ %3, %2 ], [ %8, %4 ]
  %6 = or i64 %5, %1
  %7 = cmpxchg i64* %0, i64 %5, i64 %6 seq_cst seq_cst
  %8 = extractvalue { i64, i1 } %7, 0
  %9 = icmp eq i64 %5, %8
  br i1 %9, label %10, label %4

; <label>:10:                                     ; preds = %4
  %11 = phi i64 [ %5, %4 ]
  ret i64 %11
}

; Function Attrs: nounwind
define i64 @atomic_xor_uint64(i64*, i64) #0 {
  %3 = load i64, i64* %0, align 8
  br label %4

; <label>:4:                                      ; preds = %4, %2
  %5 = phi i64 [ %3, %2 ], [ %8, %4 ]
  %6 = xor i64 %5, %1
  %7 = cmpxchg i64* %0, i64 %5, i64 %6 seq_cst seq_cst
  %8 = extractvalue { i64, i1 } %7, 0
  %9 = icmp eq i64 %5, %8
  br i1 %9, label %10, label %4

; <label>:10:                                     ; preds = %4
  %11 = phi i64 [ %5, %4 ]
  ret i64 %11
}

; Function Attrs: nounwind
define i64 @atomic_max_uint64(i64*, i64) #0 {
  %3 = load i64, i64* %0, align 8
  br label %4

; <label>:4:                                      ; preds = %4, %2
  %5 = phi i64 [ %3, %2 ], [ %9, %4 ]
  %6 = icmp ugt i64 %5, %1
  %7 = select i1 %6, i64 %5, i64 %1
  %8 = cmpxchg i64* %0, i64 %5, i64 %7 seq_cst seq_cst
  %9 = extractvalue { i64, i1 } %8, 0
  %10 = icmp eq i64 %5, %9
  br i1 %10, label %11, label %4

; <label>:11:                                     ; preds = %4
  %12 = phi i64 [ %5, %4 ]
  ret i64 %12
}

; Function Attrs: nounwind
define i64 @atomic_min_uint64(i64*, i64) #0 {
  %3 = load i64, i64* %0, align 8
  br label %4

; <label>:4:                                      ; preds = %4, %2
  %5 = phi i64 [ %3, %2 ], [ %9, %4 ]
  %6 = icmp ult i64 %5, %1
  %7 = select i1 %6, i64 %5, i64 %1
  %8 = cmpxchg i64* %0, i64 %5, i64 %7 seq_cst seq_cst
  %9 = extractvalue { i64, i1 } %8, 0
  %10 = icmp eq i64 %5, %9
  br i1 %10, label %11, label %4

; <label>:11:                                     ; preds = %4
  %12 = phi i64 [ %5, %4 ]
  ret i64 %12
}

; Function Attrs: nounwind
define i32 @atomic_inc_unsigned_global(i32 addrspace(1)*) #0 {
  %ret = atomicrmw add i32 addrspace(1)* %0, i32 1 seq_cst
  ret i32 %ret
}

; Function Attrs: nounwind
define i32 @atomic_dec_unsigned_global(i32 addrspace(1)*) #0 {
  %ret = atomicrmw sub i32 addrspace(1)* %0, i32 1 seq_cst
  ret i32 %ret
}

; Function Attrs: nounwind
define i32 @atomic_inc_unsigned_local(i32*) #0 {
  %cast = addrspacecast i32* %0 to i32 addrspace(3)*
  %ret = atomicrmw add i32 addrspace(3)* %cast, i32 1 seq_cst
  ret i32 %ret
}

; Function Attrs: nounwind
define i32 @atomic_dec_unsigned_local(i32*) #0 {
  %cast = addrspacecast i32* %0 to i32 addrspace(3)*
  %ret = atomicrmw sub i32 addrspace(3)* %cast, i32 1 seq_cst
  ret i32 %ret
}

; Function Attrs: nounwind
define i32 @atomic_inc_unsigned(i32*) #0 {
  %ret = atomicrmw add i32* %0, i32 1 seq_cst
  ret i32 %ret
}

; Function Attrs: nounwind
define i32 @atomic_dec_unsigned(i32*) #0 {
  %ret = atomicrmw sub i32* %0, i32 1 seq_cst
  ret i32 %ret
}

; Function Attrs: nounwind
define i32 @atomic_inc_int_global(i32 addrspace(1)*) #0 {
  %ret = atomicrmw add i32 addrspace(1)* %0, i32 1 seq_cst
  ret i32 %ret
}

; Function Attrs: nounwind
define i32 @atomic_dec_int_global(i32 addrspace(1)*) #0 {
  %ret = atomicrmw sub i32 addrspace(1)* %0, i32 1 seq_cst
  ret i32 %ret
}

; Function Attrs: nounwind
define i32 @atomic_inc_int_local(i32*) #0 {
  %cast = addrspacecast i32* %0 to i32 addrspace(3)*
  %ret = atomicrmw add i32 addrspace(3)* %cast, i32 1 seq_cst
  ret i32 %ret
}

; Function Attrs: nounwind
define i32 @atomic_dec_int_local(i32*) #0 {
  %cast = addrspacecast i32* %0 to i32 addrspace(3)* 
  %ret = atomicrmw sub i32 addrspace(3)* %cast, i32 1 seq_cst
  ret i32 %ret
}

; Function Attrs: nounwind
define i32 @atomic_inc_int(i32*) #0 {
  %ret = atomicrmw add i32* %0, i32 1 seq_cst
  ret i32 %ret
}

; Function Attrs: nounwind
define i32 @atomic_dec_int(i32*) #0 {
  %ret = atomicrmw sub i32* %0, i32 1 seq_cst
  ret i32 %ret
}

attributes #0 = { "disable-tail-calls"="false" "less-precise-fpmad"="false" "no-frame-pointer-elim"="true" "no-frame-pointer-elim-non-leaf" "no-infs-fp-math"="false" "no-jump-tables"="false" "no-nans-fp-math"="false" "stack-protector-buffer-size"="8" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #1 = { nounwind readnone }
attributes #2 = { convergent nounwind }
attributes #3 = { "disable-tail-calls"="false" "less-precise-fpmad"="false" "no-frame-pointer-elim"="true" "no-frame-pointer-elim-non-leaf" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "stack-protector-buffer-size"="8" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #4 = { nounwind "disable-tail-calls"="false" "less-precise-fpmad"="false" "no-frame-pointer-elim"="true" "no-frame-pointer-elim-non-leaf" "no-infs-fp-math"="false" "no-jump-tables"="false" "no-nans-fp-math"="false" "stack-protector-buffer-size"="8" "unsafe-fp-math"="false" "use-soft-float"="false" }

