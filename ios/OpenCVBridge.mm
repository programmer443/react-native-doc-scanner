//
//  OpenCVBridge.mm
//  react-native-doc-scanner
//
//  Real OpenCV C++ implementations backing OpenCVBridge.h. Two independent
//  responsibilities live here:
//   1. Per-frame quality analysis + classical document-quad fallback,
//      operating directly on a live CVPixelBuffer (see §4 in the task spec).
//   2. The post-capture OCR pipeline's image ops: perspective rectification,
//      DBNet pre/post-processing, and rec/cls crop preprocessing (§7).
//
#import "OpenCVBridge.h"
#import <UIKit/UIKit.h>

// opencv2/stitching's detail headers declare `enum { NO, ... }`, which fails
// to parse ("expected identifier") once Apple's <objc/objc.h> #defines `NO`
// to `__objc_no` — and by this point in the file it always already has been:
// CocoaPods force-injects this pod's auto-generated prefix.pch (which itself
// #imports UIKit) via a compiler `-include` flag, i.e. *before* this file's
// own text is even reached, so no reordering of #imports within this file
// can avoid the collision (confirmed the hard way — reordering alone looked
// right under a standalone `clang -fsyntax-only` check, but that check
// doesn't replicate the PCH injection, so it was a false negative). Undoing
// the macro just for this one #include, then restoring it identically to
// <objc/objc.h>'s own definition, is what actually works under the real
// Xcode build.
#undef NO
#import <opencv2/opencv.hpp>
#ifndef NO
#define NO __objc_no
#endif

#include <algorithm>
#include <cmath>
#include <vector>

#pragma mark - DSQuad / DSDetectionResult / DSQualityMetrics / DSTextBox

@implementation DSQuad
- (instancetype)initWithTopLeft:(CGPoint)topLeft
                        topRight:(CGPoint)topRight
                     bottomRight:(CGPoint)bottomRight
                      bottomLeft:(CGPoint)bottomLeft {
  if ((self = [super init])) {
    _topLeft = topLeft;
    _topRight = topRight;
    _bottomRight = bottomRight;
    _bottomLeft = bottomLeft;
  }
  return self;
}
@end

@implementation DSDetectionResult
@end

@implementation DSQualityMetrics
@end

@implementation DSTextBox
@end

#pragma mark - C++ helpers (not exposed to Swift)

namespace {

/// Converts a locked CVPixelBuffer (NV12 bi-planar YUV, full or video range,
/// or 32BGRA/32RGBA) into an owned (cloned) BGR cv::Mat, optionally cropped
/// to `roi` (full-buffer pixel coordinates; pass CGRectNull for the whole
/// frame). Returns an empty Mat for unsupported pixel formats (e.g. compressed
/// or GPU-private buffers) — callers must check `.empty()`.
cv::Mat MatFromPixelBuffer(CVPixelBufferRef pixelBuffer, CGRect roi) {
  CVReturn lockResult = CVPixelBufferLockBaseAddress(pixelBuffer, kCVPixelBufferLock_ReadOnly);
  if (lockResult != kCVReturnSuccess) {
    return cv::Mat();
  }

  OSType formatType = CVPixelBufferGetPixelFormatType(pixelBuffer);
  size_t width = CVPixelBufferGetWidth(pixelBuffer);
  size_t height = CVPixelBufferGetHeight(pixelBuffer);
  cv::Mat bgr;

  if (formatType == kCVPixelFormatType_420YpCbCr8BiPlanarFullRange ||
      formatType == kCVPixelFormatType_420YpCbCr8BiPlanarVideoRange) {
    // Camera frame processors default to bi-planar 4:2:0 YUV ('yuv' pixelFormat
    // in useFrameOutput). cv::cvtColorTwoPlane takes the Y and UV planes as
    // *separate* Mats, so we don't need them to be contiguous in memory.
    size_t yStride = CVPixelBufferGetBytesPerRowOfPlane(pixelBuffer, 0);
    void *yBase = CVPixelBufferGetBaseAddressOfPlane(pixelBuffer, 0);
    size_t uvStride = CVPixelBufferGetBytesPerRowOfPlane(pixelBuffer, 1);
    void *uvBase = CVPixelBufferGetBaseAddressOfPlane(pixelBuffer, 1);
    if (!yBase || !uvBase) {
      CVPixelBufferUnlockBaseAddress(pixelBuffer, kCVPixelBufferLock_ReadOnly);
      return cv::Mat();
    }
    cv::Mat yMat((int)height, (int)width, CV_8UC1, yBase, yStride);
    cv::Mat uvMat((int)(height / 2), (int)(width / 2), CV_8UC2, uvBase, uvStride);
    // KNOWN LIMITATION: stock OpenCV's COLOR_YUV2BGR_NV12 always applies the
    // full-range conversion matrix; there's no built-in "video range" (16-235
    // luma / 16-240 chroma) variant. For VideoRange buffers this yields
    // slightly compressed contrast (not clipped/incorrect, just not
    // full-range-expanded) rather than a hard color bug. This only affects
    // the *look* of the BGR Mat used for blur/brightness/glare/contour
    // analysis, not correctness of those measurements (Canny/Laplacian/mean
    // are not sensitive to this level of contrast compression), so it's an
    // accepted simplification rather than a fixed-up special case.
    cv::cvtColorTwoPlane(yMat, uvMat, bgr, cv::COLOR_YUV2BGR_NV12);
  } else if (formatType == kCVPixelFormatType_32BGRA) {
    size_t stride = CVPixelBufferGetBytesPerRow(pixelBuffer);
    void *base = CVPixelBufferGetBaseAddress(pixelBuffer);
    if (!base) {
      CVPixelBufferUnlockBaseAddress(pixelBuffer, kCVPixelBufferLock_ReadOnly);
      return cv::Mat();
    }
    cv::Mat bgra((int)height, (int)width, CV_8UC4, base, stride);
    cv::cvtColor(bgra, bgr, cv::COLOR_BGRA2BGR);
  } else if (formatType == kCVPixelFormatType_32RGBA) {
    size_t stride = CVPixelBufferGetBytesPerRow(pixelBuffer);
    void *base = CVPixelBufferGetBaseAddress(pixelBuffer);
    if (!base) {
      CVPixelBufferUnlockBaseAddress(pixelBuffer, kCVPixelBufferLock_ReadOnly);
      return cv::Mat();
    }
    cv::Mat rgba((int)height, (int)width, CV_8UC4, base, stride);
    cv::cvtColor(rgba, bgr, cv::COLOR_RGBA2BGR);
  } else {
    // Unsupported/opaque format (e.g. a GPU-private buffer). Nothing sane to
    // do on the CPU here.
    CVPixelBufferUnlockBaseAddress(pixelBuffer, kCVPixelBufferLock_ReadOnly);
    return cv::Mat();
  }

  cv::Mat result;
  if (!CGRectIsNull(roi) && !CGRectIsEmpty(roi)) {
    cv::Rect r((int)roi.origin.x, (int)roi.origin.y, (int)roi.size.width, (int)roi.size.height);
    cv::Rect bounds(0, 0, bgr.cols, bgr.rows);
    r = r & bounds;
    result = (r.width > 0 && r.height > 0) ? bgr(r).clone() : bgr.clone();
  } else {
    result = bgr.clone();
  }

  CVPixelBufferUnlockBaseAddress(pixelBuffer, kCVPixelBufferLock_ReadOnly);
  return result;
}

/// Downscales `src` (keeping aspect ratio) so its longer side is at most
/// `maxLongSide`. Returns `src` unchanged if it's already small enough.
cv::Mat DownscaleKeepingAspect(const cv::Mat &src, int maxLongSide) {
  if (src.empty()) return src;
  int longSide = std::max(src.cols, src.rows);
  if (longSide <= maxLongSide) return src;
  double scale = (double)maxLongSide / (double)longSide;
  cv::Mat dst;
  cv::resize(src, dst, cv::Size(), scale, scale, cv::INTER_AREA);
  return dst;
}

/// Orders 4 arbitrary points into [topLeft, topRight, bottomRight, bottomLeft]
/// using the standard doc-scanner sum/difference heuristic: top-left has the
/// smallest x+y, bottom-right the largest x+y, top-right the smallest y-x,
/// bottom-left the largest y-x. Assumes the quad isn't rotated so far within
/// its own plane that this heuristic breaks down (reasonable for a
/// guided-capture document photo).
std::vector<cv::Point2f> OrderQuadPoints(const std::vector<cv::Point2f> &pts) {
  std::vector<double> sum(pts.size()), diff(pts.size());
  for (size_t i = 0; i < pts.size(); i++) {
    sum[i] = pts[i].x + pts[i].y;
    diff[i] = pts[i].y - pts[i].x;
  }
  int tlIdx = (int)(std::min_element(sum.begin(), sum.end()) - sum.begin());
  int brIdx = (int)(std::max_element(sum.begin(), sum.end()) - sum.begin());
  int trIdx = (int)(std::min_element(diff.begin(), diff.end()) - diff.begin());
  int blIdx = (int)(std::max_element(diff.begin(), diff.end()) - diff.begin());
  return {pts[tlIdx], pts[trIdx], pts[brIdx], pts[blIdx]};
}

/// Interior angle deviation from 90°, in degrees, at vertex `b` formed by
/// edges b->a and b->c.
double AngleDeviationFrom90(const cv::Point2f &a, const cv::Point2f &b, const cv::Point2f &c) {
  cv::Point2f v1 = a - b;
  cv::Point2f v2 = c - b;
  double dot = v1.x * v2.x + v1.y * v2.y;
  double mag = cv::norm(v1) * cv::norm(v2);
  if (mag < 1e-6) return 90.0;
  double cosAngle = std::max(-1.0, std::min(1.0, dot / mag));
  double angleDeg = std::acos(cosAngle) * 180.0 / CV_PI;
  return std::abs(angleDeg - 90.0);
}

/// 0-1 "how rectangular is this quad" score: 1.0 = all four corners are
/// exactly 90°, 0.0 = at least one corner is 45°+ away from 90°.
double RectangularityScore(const std::vector<cv::Point2f> &q) {
  double d0 = AngleDeviationFrom90(q[3], q[0], q[1]);
  double d1 = AngleDeviationFrom90(q[0], q[1], q[2]);
  double d2 = AngleDeviationFrom90(q[1], q[2], q[3]);
  double d3 = AngleDeviationFrom90(q[2], q[3], q[0]);
  double maxDeviation = std::max({d0, d1, d2, d3});
  return std::max(0.0, std::min(1.0, 1.0 - maxDeviation / 45.0));
}

/// Standard iOS UIImage -> BGR cv::Mat conversion. Draws through
/// UIGraphicsImageRenderer (rather than reading `image.CGImage` directly) so
/// the result respects `UIImage.imageOrientation` (EXIF) — reading CGImage
/// directly would silently ignore any orientation tag and yield
/// sideways/upside-down pixels for photos whose pixel data isn't already
/// physically upright.
cv::Mat MatFromUIImage(UIImage *image) {
  CGFloat width = image.size.width;
  CGFloat height = image.size.height;
  if (width <= 0 || height <= 0) return cv::Mat();

  UIGraphicsImageRendererFormat *format = [UIGraphicsImageRendererFormat preferredFormat];
  format.opaque = YES;
  format.scale = 1.0;
  UIGraphicsImageRenderer *renderer =
      [[UIGraphicsImageRenderer alloc] initWithSize:CGSizeMake(width, height) format:format];
  UIImage *normalized = [renderer imageWithActions:^(UIGraphicsImageRendererContext *_Nonnull ctx) {
    [image drawInRect:CGRectMake(0, 0, width, height)];
  }];

  CGImageRef cgImage = normalized.CGImage;
  if (!cgImage) return cv::Mat();

  int cols = (int)CGImageGetWidth(cgImage);
  int rows = (int)CGImageGetHeight(cgImage);
  cv::Mat rgba(rows, cols, CV_8UC4);

  CGColorSpaceRef colorSpace = CGColorSpaceCreateDeviceRGB();
  CGContextRef contextRef = CGBitmapContextCreate(
      rgba.data, cols, rows, 8, rgba.step[0], colorSpace,
      kCGImageAlphaNoneSkipLast | kCGBitmapByteOrderDefault);
  CGColorSpaceRelease(colorSpace);
  if (!contextRef) return cv::Mat();

  CGContextDrawImage(contextRef, CGRectMake(0, 0, cols, rows), cgImage);
  CGContextRelease(contextRef);

  cv::Mat bgr;
  cv::cvtColor(rgba, bgr, cv::COLOR_RGBA2BGR);
  return bgr;
}

/// Crops+deskews `quad` out of `src` via a perspective warp sized to the
/// quad's own average width/height (RapidOCR's `get_rotate_crop_image`
/// technique), then rotates 90° if the crop came out "tall" (rotated
/// vertical text line).
cv::Mat GetRotateCropImage(const cv::Mat &src, DSQuad *quad) {
  if (src.empty() || quad == nil) return cv::Mat();

  std::vector<cv::Point2f> pts = {
      cv::Point2f((float)quad.topLeft.x, (float)quad.topLeft.y),
      cv::Point2f((float)quad.topRight.x, (float)quad.topRight.y),
      cv::Point2f((float)quad.bottomRight.x, (float)quad.bottomRight.y),
      cv::Point2f((float)quad.bottomLeft.x, (float)quad.bottomLeft.y),
  };

  double widthTop = cv::norm(pts[0] - pts[1]);
  double widthBottom = cv::norm(pts[3] - pts[2]);
  double heightLeft = cv::norm(pts[0] - pts[3]);
  double heightRight = cv::norm(pts[1] - pts[2]);
  int dstW = std::max(1, (int)std::round(std::max(widthTop, widthBottom)));
  int dstH = std::max(1, (int)std::round(std::max(heightLeft, heightRight)));

  std::vector<cv::Point2f> dstPts = {
      cv::Point2f(0, 0),
      cv::Point2f((float)dstW, 0),
      cv::Point2f((float)dstW, (float)dstH),
      cv::Point2f(0, (float)dstH),
  };

  cv::Mat m = cv::getPerspectiveTransform(pts, dstPts);
  cv::Mat crop;
  cv::warpPerspective(src, crop, m, cv::Size(dstW, dstH), cv::INTER_LINEAR, cv::BORDER_REPLICATE);

  if (!crop.empty() && (double)crop.rows >= (double)crop.cols * 1.5) {
    cv::Mat rotated;
    cv::rotate(crop, rotated, cv::ROTATE_90_CLOCKWISE);
    return rotated;
  }
  return crop;
}

/// Fills an RGB float32 NCHW buffer, normalised as `(pixel/255 - mean[c]) /
/// std[c]`, from a BGR 8-bit Mat (converted to RGB internally).
NSData *NCHWFromBGR(const cv::Mat &bgr, const float mean[3], const float stdv[3]) {
  cv::Mat rgb;
  cv::cvtColor(bgr, rgb, cv::COLOR_BGR2RGB);
  int h = rgb.rows, w = rgb.cols;
  NSMutableData *data = [NSMutableData dataWithLength:sizeof(float) * 3 * h * w];
  float *dst = (float *)data.mutableBytes;
  int plane = h * w;
  for (int y = 0; y < h; y++) {
    const uint8_t *row = rgb.ptr<uint8_t>(y);
    for (int x = 0; x < w; x++) {
      int idx = y * w + x;
      for (int c = 0; c < 3; c++) {
        float v = row[x * 3 + c] / 255.0f;
        dst[c * plane + idx] = (v - mean[c]) / stdv[c];
      }
    }
  }
  return data;
}

/// Fills a BGR float32 NCHW buffer with RAW pixel values (0-255, no /255
/// normalisation, no mean/std, no BGR->RGB conversion) from a BGR 8-bit Mat —
/// YuNet's expected input. Deliberately NOT built on top of `NCHWFromBGR`
/// above: that helper always divides by 255 and swaps to RGB, neither of
/// which YuNet wants (see OpenCVBridge.h's doc comment on
/// `preprocessFaceDetectorInputWithPixelBuffer:`).
NSData *NCHWFromBGRRaw(const cv::Mat &bgr) {
  int h = bgr.rows, w = bgr.cols;
  NSMutableData *data = [NSMutableData dataWithLength:sizeof(float) * 3 * h * w];
  float *dst = (float *)data.mutableBytes;
  int plane = h * w;
  for (int y = 0; y < h; y++) {
    const uint8_t *row = bgr.ptr<uint8_t>(y);
    for (int x = 0; x < w; x++) {
      int idx = y * w + x;
      for (int c = 0; c < 3; c++) {
        dst[c * plane + idx] = (float)row[x * 3 + c];
      }
    }
  }
  return data;
}

}  // namespace

#pragma mark - OpenCVBridge

@interface OpenCVBridge () {
  cv::Mat _previousGray;  // per-frame motion-detection state
  cv::Mat _captureMat;    // current capture-pipeline working image (BGR)
}
@end

@implementation OpenCVBridge

#pragma mark Quality + classical detection

- (DSQualityMetrics *)analyzeQualityWithPixelBuffer:(CVPixelBufferRef)pixelBuffer roi:(CGRect)roi {
  DSQualityMetrics *metrics = [DSQualityMetrics new];
  cv::Mat bgr = MatFromPixelBuffer(pixelBuffer, roi);
  if (bgr.empty()) {
    return metrics;
  }

  // Quality metrics run on a ~480px-long-edge downscale: full-resolution
  // Laplacian/histogram analysis is unnecessary for these coarse signals and
  // would cost real per-frame latency at 1080p+ (see docs/TROUBLESHOOTING.md's
  // perf notes).
  cv::Mat working = DownscaleKeepingAspect(bgr, 480);
  cv::Mat gray;
  cv::cvtColor(working, gray, cv::COLOR_BGR2GRAY);

  // Blur: Laplacian variance.
  cv::Mat laplacian;
  cv::Laplacian(gray, laplacian, CV_64F);
  cv::Scalar mean, stddev;
  cv::meanStdDev(laplacian, mean, stddev);
  metrics.blurScore = stddev[0] * stddev[0];

  // Brightness: mean luminance.
  metrics.brightness = cv::mean(gray)[0];

  // Glare: fraction of near-blown-out pixels.
  cv::Mat thresholded;
  cv::threshold(gray, thresholded, 250, 255, cv::THRESH_BINARY);
  double totalPixels = (double)thresholded.rows * (double)thresholded.cols;
  metrics.glareRatio = totalPixels > 0 ? cv::countNonZero(thresholded) / totalPixels : 0.0;

  // Motion: mean abs diff vs. the previous call's downscaled grayscale frame.
  if (!_previousGray.empty() && _previousGray.size() == gray.size()) {
    cv::Mat diff;
    cv::absdiff(_previousGray, gray, diff);
    metrics.motionScore = std::min(1.0, cv::mean(diff)[0] / 255.0);
  } else {
    metrics.motionScore = 0.0;
  }
  _previousGray = gray.clone();

  return metrics;
}

- (nullable DSDetectionResult *)detectDocumentQuadWithPixelBuffer:(CVPixelBufferRef)pixelBuffer {
  cv::Mat bgrFull = MatFromPixelBuffer(pixelBuffer, CGRectNull);
  if (bgrFull.empty()) return nil;

  // The classical contour fallback needs more spatial detail than the 256x256
  // ONNX input to find accurate corners, but we still downscale for speed —
  // 640px long edge is a reasonable accuracy/perf balance for
  // cv::findContours + cv::approxPolyDP on a live camera feed.
  int longSide = std::max(bgrFull.cols, bgrFull.rows);
  double scale = longSide > 640 ? (640.0 / (double)longSide) : 1.0;
  cv::Mat working;
  if (scale < 1.0) {
    cv::resize(bgrFull, working, cv::Size(), scale, scale, cv::INTER_AREA);
  } else {
    working = bgrFull;
  }

  cv::Mat gray, blurred, edges;
  cv::cvtColor(working, gray, cv::COLOR_BGR2GRAY);
  cv::GaussianBlur(gray, blurred, cv::Size(5, 5), 0);
  cv::Canny(blurred, edges, 50, 150);
  cv::dilate(edges, edges, cv::Mat(), cv::Point(-1, -1), 1);

  std::vector<std::vector<cv::Point>> contours;
  cv::findContours(edges, contours, cv::RETR_LIST, cv::CHAIN_APPROX_SIMPLE);

  double frameArea = (double)working.cols * (double)working.rows;
  double bestArea = 0;
  std::vector<cv::Point> bestQuad;

  for (auto &contour : contours) {
    double area = cv::contourArea(contour);
    // Ignore tiny contours (noise) and implausibly large ones (the frame
    // border itself).
    if (area < frameArea * 0.1 || area > frameArea * 0.98) continue;

    double peri = cv::arcLength(contour, true);
    std::vector<cv::Point> approx;
    cv::approxPolyDP(contour, approx, 0.02 * peri, true);

    if (approx.size() == 4 && cv::isContourConvex(approx) && area > bestArea) {
      bestArea = area;
      bestQuad = approx;
    }
  }

  if (bestQuad.empty()) return nil;

  std::vector<cv::Point2f> floatPts(bestQuad.begin(), bestQuad.end());
  std::vector<cv::Point2f> ordered = OrderQuadPoints(floatPts);

  double angleScore = RectangularityScore(ordered);
  double areaRatio = bestArea / frameArea;
  // A plausible document fills a meaningful-but-not-total fraction of the
  // frame; penalise both "too small" and "too close/edge-to-edge" cases.
  double areaScore = 1.0 - std::min(1.0, std::abs(areaRatio - 0.55) / 0.55);
  double confidence = std::max(0.0, std::min(1.0, 0.5 * angleScore + 0.5 * areaScore));

  double invScale = 1.0 / scale;
  DSQuad *quad =
      [[DSQuad alloc] initWithTopLeft:CGPointMake(ordered[0].x * invScale, ordered[0].y * invScale)
                              topRight:CGPointMake(ordered[1].x * invScale, ordered[1].y * invScale)
                           bottomRight:CGPointMake(ordered[2].x * invScale, ordered[2].y * invScale)
                            bottomLeft:CGPointMake(ordered[3].x * invScale, ordered[3].y * invScale)];

  DSDetectionResult *result = [DSDetectionResult new];
  result.detected = YES;
  result.quad = quad;
  result.confidence = confidence;
  return result;
}

- (nullable NSData *)preprocessDetectorInputWithPixelBuffer:(CVPixelBufferRef)pixelBuffer
                                                  targetSize:(int)targetSize
                                                      scaleX:(double *)outScaleX
                                                      scaleY:(double *)outScaleY {
  cv::Mat bgr = MatFromPixelBuffer(pixelBuffer, CGRectNull);
  if (bgr.empty()) return nil;

  // Plain (aspect-distorting) resize straight to targetSize x targetSize —
  // deliberately NOT a letterbox. DocAligner's own preprocessing
  // (`cb.imresize(img, size=img_size_infer)`) squashes to a fixed square
  // with no padding, so a letterboxed input would feed the model a
  // different distribution than it was trained/exported on, and — more
  // visibly — the inverse coordinate mapping back to buffer space would be
  // systematically wrong for any non-square frame (i.e. every camera frame).
  double scaleX = (double)targetSize / bgr.cols;
  double scaleY = (double)targetSize / bgr.rows;
  cv::Mat resized;
  cv::resize(bgr, resized, cv::Size(targetSize, targetSize), 0, 0, cv::INTER_LINEAR);

  // Detector input is a plain 0-1 normalisation (no ImageNet mean/std) —
  // confirmed against DocAligner's actual heatmap_reg/infer.py (`img / 255.`).
  static const float mean[3] = {0.0f, 0.0f, 0.0f};
  static const float stdv[3] = {1.0f, 1.0f, 1.0f};
  NSData *data = NCHWFromBGR(resized, mean, stdv);

  if (outScaleX) *outScaleX = scaleX;
  if (outScaleY) *outScaleY = scaleY;
  return data;
}

- (nullable NSData *)preprocessFaceDetectorInputWithPixelBuffer:(CVPixelBufferRef)pixelBuffer
                                                       targetSize:(int)targetSize
                                                           scaleX:(double *)outScaleX
                                                           scaleY:(double *)outScaleY {
  cv::Mat bgr = MatFromPixelBuffer(pixelBuffer, CGRectNull);
  if (bgr.empty()) return nil;

  // Plain (aspect-distorting) squash-resize, same non-letterboxed convention
  // as preprocessDetectorInputWithPixelBuffer: above — 320 is already a
  // multiple of 32 so no additional stride padding is needed beyond this
  // square resize itself.
  double scaleX = (double)targetSize / bgr.cols;
  double scaleY = (double)targetSize / bgr.rows;
  cv::Mat resized;
  cv::resize(bgr, resized, cv::Size(targetSize, targetSize), 0, 0, cv::INTER_LINEAR);

  // YuNet input: raw 0-255 float BGR (no normalisation) — see NCHWFromBGRRaw.
  NSData *data = NCHWFromBGRRaw(resized);

  if (outScaleX) *outScaleX = scaleX;
  if (outScaleY) *outScaleY = scaleY;
  return data;
}

- (nullable NSData *)preprocessClassifierInputWithPixelBuffer:(CVPixelBufferRef)pixelBuffer
                                                           roi:(CGRect)roi
                                                    targetSize:(int)targetSize {
  cv::Mat bgr = MatFromPixelBuffer(pixelBuffer, roi);
  if (bgr.empty()) return nil;

  cv::Mat resized;
  cv::resize(bgr, resized, cv::Size(targetSize, targetSize), 0, 0, cv::INTER_LINEAR);

  // ImageNet mean/std — matches the standard preprocessing for an
  // ImageNet-pretrained backbone (MobileNetV3/EfficientNet-Lite). Update
  // alongside whatever backbone docs/MODEL_TRAINING.md §6 actually documents.
  static const float mean[3] = {0.485f, 0.456f, 0.406f};
  static const float stdv[3] = {0.229f, 0.224f, 0.225f};
  return NCHWFromBGR(resized, mean, stdv);
}

#pragma mark Capture pipeline

- (CGSize)loadCaptureImageAtPath:(NSString *)imagePath {
  std::string path([imagePath UTF8String]);
  // cv::imread applies EXIF orientation by default (IMREAD_IGNORE_ORIENTATION
  // is *not* set), so the resulting Mat's pixels are already physically
  // upright for JPEGs (the common case for a VisionCamera photo capture).
  cv::Mat img = cv::imread(path, cv::IMREAD_COLOR);
  if (img.empty()) {
    // cv::imread doesn't support every format some devices may hand us
    // (notably HEIC on some OpenCV builds) — fall back to UIImage/ImageIO,
    // which does, and also respects imageOrientation (see MatFromUIImage).
    NSData *fileData = [NSData dataWithContentsOfFile:imagePath];
    if (fileData) {
      UIImage *uiImage = [UIImage imageWithData:fileData];
      if (uiImage) {
        img = MatFromUIImage(uiImage);
      }
    }
  }

  if (img.empty()) {
    _captureMat = cv::Mat();
    return CGSizeZero;
  }
  _captureMat = img;
  return CGSizeMake(img.cols, img.rows);
}

- (CGSize)currentImageSize {
  if (_captureMat.empty()) return CGSizeZero;
  return CGSizeMake(_captureMat.cols, _captureMat.rows);
}

- (BOOL)rectifyCaptureImageWithQuad:(DSQuad *)quad outputPath:(NSString *)outputPath {
  if (_captureMat.empty()) return NO;
  cv::Mat rectified = GetRotateCropImage(_captureMat, quad);
  if (rectified.empty()) return NO;

  _captureMat = rectified;

  std::string outPath([outputPath UTF8String]);
  std::vector<int> params = {cv::IMWRITE_JPEG_QUALITY, 92};
  return cv::imwrite(outPath, rectified, params);
}

- (nullable NSData *)preprocessOcrDetInputWithMaxSide:(int)maxSide
                                              outWidth:(int *)outWidth
                                             outHeight:(int *)outHeight {
  if (_captureMat.empty()) return nil;
  int h = _captureMat.rows, w = _captureMat.cols;
  double ratio = 1.0;
  if (std::max(h, w) > maxSide) {
    ratio = (double)maxSide / (double)std::max(h, w);
  }
  int resizeH = std::max(32, (int)std::round((double)(h * ratio) / 32.0) * 32);
  int resizeW = std::max(32, (int)std::round((double)(w * ratio) / 32.0) * 32);

  cv::Mat resized;
  cv::resize(_captureMat, resized, cv::Size(resizeW, resizeH));

  // PP-OCR det normalisation: ImageNet mean/std, applied to a 0-1 scaled image.
  static const float mean[3] = {0.485f, 0.456f, 0.406f};
  static const float stdv[3] = {0.229f, 0.224f, 0.225f};
  NSData *data = NCHWFromBGR(resized, mean, stdv);

  if (outWidth) *outWidth = resizeW;
  if (outHeight) *outHeight = resizeH;
  return data;
}

- (NSArray<DSTextBox *> *)textBoxesFromProbabilityMap:(NSData *)probabilityMap
                                              mapWidth:(int)mapWidth
                                             mapHeight:(int)mapHeight
                                             threshold:(double)threshold {
  NSMutableArray<DSTextBox *> *results = [NSMutableArray array];
  if (_captureMat.empty() || mapWidth <= 0 || mapHeight <= 0) return results;
  if (probabilityMap.length < sizeof(float) * (size_t)mapWidth * (size_t)mapHeight) return results;

  cv::Mat prob((int)mapHeight, (int)mapWidth, CV_32FC1, (void *)probabilityMap.bytes);
  cv::Mat mask8u;
  prob.convertTo(mask8u, CV_8UC1, 255.0);
  cv::Mat binary;
  cv::threshold(mask8u, binary, threshold * 255.0, 255, cv::THRESH_BINARY);

  std::vector<std::vector<cv::Point>> contours;
  cv::findContours(binary, contours, cv::RETR_LIST, cv::CHAIN_APPROX_SIMPLE);

  double scaleX = (double)_captureMat.cols / (double)mapWidth;
  double scaleY = (double)_captureMat.rows / (double)mapHeight;

  std::vector<std::pair<double, DSTextBox *>> boxesWithY;

  for (auto &contour : contours) {
    if (contour.size() < 3) continue;
    double area = cv::contourArea(contour);
    if (area < 4) continue;  // ignore specks

    cv::RotatedRect rect = cv::minAreaRect(contour);

    // DB models are trained to predict a *shrunk* text region; the standard
    // post-process expands ("unclips") the box outward using Vatti clipping
    // by `area * unclip_ratio / perimeter`. We approximate that with a fixed
    // ~1.5x scale-up around the rect's center — a well-known simplification
    // (also used by several lightweight RapidOCR ports) when the full
    // polygon-offset unclip isn't implemented.
    const double kUnclipScale = 1.5;
    cv::Size2f size = rect.size;
    size.width *= kUnclipScale;
    size.height *= kUnclipScale;
    cv::RotatedRect expanded(rect.center, size, rect.angle);

    cv::Point2f pts[4];
    expanded.points(pts);

    double wA = cv::norm(pts[0] - pts[1]);
    double wB = cv::norm(pts[2] - pts[3]);
    double hA = cv::norm(pts[1] - pts[2]);
    double hB = cv::norm(pts[3] - pts[0]);
    double boxW = (wA + wB) / 2.0 * scaleX;
    double boxH = (hA + hB) / 2.0 * scaleY;
    if (boxW < 4 || boxH < 4) continue;

    std::vector<cv::Point2f> scaledPts(4);
    for (int i = 0; i < 4; i++) {
      scaledPts[i] = cv::Point2f((float)(pts[i].x * scaleX), (float)(pts[i].y * scaleY));
    }
    // Re-order using the same sum/diff heuristic as the classical detector,
    // for consistent downstream cropping regardless of minAreaRect's
    // internal point-ordering convention.
    std::vector<cv::Point2f> ordered = OrderQuadPoints(scaledPts);

    DSQuad *quad = [[DSQuad alloc] initWithTopLeft:CGPointMake(ordered[0].x, ordered[0].y)
                                            topRight:CGPointMake(ordered[1].x, ordered[1].y)
                                         bottomRight:CGPointMake(ordered[2].x, ordered[2].y)
                                          bottomLeft:CGPointMake(ordered[3].x, ordered[3].y)];

    double minX = std::min({ordered[0].x, ordered[1].x, ordered[2].x, ordered[3].x});
    double maxX = std::max({ordered[0].x, ordered[1].x, ordered[2].x, ordered[3].x});
    double minY = std::min({ordered[0].y, ordered[1].y, ordered[2].y, ordered[3].y});
    double maxY = std::max({ordered[0].y, ordered[1].y, ordered[2].y, ordered[3].y});

    DSTextBox *box = [DSTextBox new];
    box.quad = quad;
    box.boundingBox = CGRectMake(minX, minY, maxX - minX, maxY - minY);

    double vCenter = (minY + maxY) / 2.0;
    boxesWithY.push_back({vCenter, box});
  }

  std::sort(boxesWithY.begin(), boxesWithY.end(),
            [](const auto &a, const auto &b) { return a.first < b.first; });
  for (auto &p : boxesWithY) {
    [results addObject:p.second];
  }
  return results;
}

- (nullable NSData *)preprocessRecInputForBox:(DSTextBox *)box
                                     rotate180:(BOOL)rotate180
                                  targetHeight:(int)targetHeight
                                      outWidth:(int *)outWidth {
  if (_captureMat.empty()) return nil;
  cv::Mat crop = GetRotateCropImage(_captureMat, box.quad);
  if (crop.empty()) return nil;
  if (rotate180) {
    cv::rotate(crop, crop, cv::ROTATE_180);
  }

  double ratio = (double)targetHeight / (double)crop.rows;
  int targetWidth = std::max(1, (int)std::round(crop.cols * ratio));
  // Clamp absurdly wide lines (e.g. a mis-detected full-row box) so the rec
  // model isn't fed an unreasonably large tensor.
  targetWidth = std::min(targetWidth, 1600);

  cv::Mat resized;
  cv::resize(crop, resized, cv::Size(targetWidth, targetHeight));

  // PP-OCR rec normalisation: maps 0-1 pixel values to [-1, 1].
  static const float mean[3] = {0.5f, 0.5f, 0.5f};
  static const float stdv[3] = {0.5f, 0.5f, 0.5f};
  NSData *data = NCHWFromBGR(resized, mean, stdv);

  if (outWidth) *outWidth = targetWidth;
  return data;
}

- (nullable NSData *)preprocessClsInputForBox:(DSTextBox *)box clsWidth:(int)clsWidth clsHeight:(int)clsHeight {
  if (_captureMat.empty()) return nil;
  cv::Mat crop = GetRotateCropImage(_captureMat, box.quad);
  if (crop.empty()) return nil;

  double ratio = (double)crop.cols / (double)crop.rows;
  int resizedW = std::min(clsWidth, std::max(1, (int)std::ceil(clsHeight * ratio)));
  cv::Mat resized;
  cv::resize(crop, resized, cv::Size(resizedW, clsHeight));

  cv::Mat canvas(clsHeight, clsWidth, CV_8UC3, cv::Scalar(0, 0, 0));
  resized.copyTo(canvas(cv::Rect(0, 0, resized.cols, resized.rows)));

  static const float mean[3] = {0.5f, 0.5f, 0.5f};
  static const float stdv[3] = {0.5f, 0.5f, 0.5f};
  return NCHWFromBGR(canvas, mean, stdv);
}

@end
