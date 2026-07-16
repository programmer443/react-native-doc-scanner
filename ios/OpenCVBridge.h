//
//  OpenCVBridge.h
//  react-native-doc-scanner
//
//  Objective-C surface over the real OpenCV C++ operations used by both the
//  per-frame analyzer (HybridDocScanner.analyzeFrame) and the post-capture
//  OCR pipeline (HybridDocScanner.captureAndExtract). The header is plain
//  Objective-C (no C++ types) so it's directly visible to Swift from within
//  this pod's module (see `DEFINES_MODULE => YES` in the podspec) — all
//  `cv::Mat`/C++ usage is confined to OpenCVBridge.mm.
//
//  Every method has an explicit NS_SWIFT_NAME so the Swift call sites in
//  ONNXInference.swift / HybridDocScanner.swift are pinned and unambiguous.
//

#import <Foundation/Foundation.h>
#import <CoreGraphics/CoreGraphics.h>
#import <CoreVideo/CoreVideo.h>

NS_ASSUME_NONNULL_BEGIN

/// Four corners of a document/text-line quadrilateral, clockwise from
/// top-left, in the pixel space of whatever buffer/image they were computed
/// against (callers are responsible for any orientation normalisation).
NS_SWIFT_NAME(DSQuad)
@interface DSQuad : NSObject
@property (nonatomic, readonly) CGPoint topLeft;
@property (nonatomic, readonly) CGPoint topRight;
@property (nonatomic, readonly) CGPoint bottomRight;
@property (nonatomic, readonly) CGPoint bottomLeft;
- (instancetype)initWithTopLeft:(CGPoint)topLeft
                        topRight:(CGPoint)topRight
                     bottomRight:(CGPoint)bottomRight
                      bottomLeft:(CGPoint)bottomLeft;
@end

/// Result of the classical-CV contour-based document detector fallback.
NS_SWIFT_NAME(DSDetectionResult)
@interface DSDetectionResult : NSObject
@property (nonatomic, assign) BOOL detected;
@property (nonatomic, strong, nullable) DSQuad *quad;
@property (nonatomic, assign) double confidence;
@end

/// Per-frame OpenCV quality metrics (blur/brightness/glare/motion). Distance
/// ratio, perspective skew, and out-of-frame ratio are pure quad geometry and
/// are computed natively in Swift instead (see HybridDocScanner.swift).
NS_SWIFT_NAME(DSQualityMetrics)
@interface DSQualityMetrics : NSObject
@property (nonatomic, assign) double blurScore;
@property (nonatomic, assign) double brightness;
@property (nonatomic, assign) double glareRatio;
@property (nonatomic, assign) double motionScore;
@end

/// One rotated text-line box found by the DBNet-style det-model post-process,
/// in the *current capture image's* full pixel coordinate space.
NS_SWIFT_NAME(DSTextBox)
@interface DSTextBox : NSObject
@property (nonatomic, strong) DSQuad *quad;
@property (nonatomic, assign) CGRect boundingBox;
@end

/// Stateful OpenCV helper. One instance should back one long-lived per-frame
/// analyzer (it remembers the previous frame's grayscale buffer for motion
/// scoring), and a *separate*, short-lived instance should back one
/// `captureAndExtract` call (it remembers the current capture image, which
/// gets replaced in place by perspective rectification).
NS_SWIFT_NAME(OpenCVBridge)
@interface OpenCVBridge : NSObject

#pragma mark - Per-frame quality + detection (CVPixelBuffer-based)

/// Runs blur/brightness/glare/motion analysis on a (locked or lockable)
/// CVPixelBuffer. `roi` is in full-buffer pixel coordinates; pass `CGRectNull`
/// to analyse the whole frame. Motion score compares against the previous
/// call's downscaled grayscale buffer, stored as instance state.
- (DSQualityMetrics *)analyzeQualityWithPixelBuffer:(CVPixelBufferRef)pixelBuffer
                                                 roi:(CGRect)roi
    NS_SWIFT_NAME(analyzeQuality(withPixelBuffer:roi:));

/// Classical OpenCV document-quad fallback: grayscale -> GaussianBlur -> Canny
/// -> findContours -> approxPolyDP, keeping the largest plausible 4-point
/// convex contour. Returns nil if no plausible quad was found. Quad
/// coordinates are in full pixel-buffer space.
- (nullable DSDetectionResult *)detectDocumentQuadWithPixelBuffer:(CVPixelBufferRef)pixelBuffer
    NS_SWIFT_NAME(detectDocumentQuad(withPixelBuffer:));

/// Resizes a CVPixelBuffer directly to a square `targetSize x targetSize` RGB
/// float32 NCHW tensor (0-1 normalised), ready for an ONNX Runtime input.
/// Deliberately a plain (aspect-distorting) resize, NOT a letterbox — matches
/// DocAligner's own `heatmap_reg/infer.py` preprocessing (`cb.imresize(img,
/// size=img_size_infer)`, no padding). Reports the resulting per-axis scale
/// so model-space output coordinates can be mapped back to full
/// pixel-buffer space via `bufferCoord = modelCoord / scale`.
- (nullable NSData *)preprocessDetectorInputWithPixelBuffer:(CVPixelBufferRef)pixelBuffer
                                                  targetSize:(int)targetSize
                                                      scaleX:(double *)outScaleX
                                                      scaleY:(double *)outScaleY
    NS_SWIFT_NAME(preprocessDetectorInput(withPixelBuffer:targetSize:scaleX:scaleY:));

/// Resizes a CVPixelBuffer directly to a square `targetSize x targetSize` BGR
/// float32 NCHW tensor of RAW pixel values: no /255 normalisation, no
/// mean/std, and channel order preserved as BGR (NOT converted to RGB). This
/// matches YuNet's own preprocessing (OpenCV's `face_detect.cpp` calls
/// `dnn::blobFromImage(pad_image)` with no scalefactor/mean and no
/// `swapRB`) — deliberately different from `preprocessDetectorInputWithPixelBuffer:`
/// above (which 0-1 normalises and converts to RGB for DocAligner). Same
/// plain (aspect-distorting) squash-resize convention as that method, for the
/// same reason — see its doc comment. Reports the resulting per-axis scale so
/// model-space output coordinates can be mapped back to full pixel-buffer
/// space via `bufferCoord = modelCoord / scale`.
- (nullable NSData *)preprocessFaceDetectorInputWithPixelBuffer:(CVPixelBufferRef)pixelBuffer
                                                       targetSize:(int)targetSize
                                                           scaleX:(double *)outScaleX
                                                           scaleY:(double *)outScaleY
    NS_SWIFT_NAME(preprocessFaceDetectorInput(withPixelBuffer:targetSize:scaleX:scaleY:));

/// Crops to `roi` (full-buffer pixel coordinates; pass `CGRectNull` to use the
/// whole frame — same convention as `analyzeQualityWithPixelBuffer:roi:`) and
/// squash-resizes (plain aspect-distorting resize, same non-letterboxed
/// convention as `preprocessDetectorInputWithPixelBuffer:`) to a square
/// `targetSize x targetSize` RGB float32 NCHW tensor, ImageNet mean/std
/// normalised (`mean=[0.485,0.456,0.406]`, `std=[0.229,0.224,0.225]`) — the
/// standard input convention for an ImageNet-pretrained backbone
/// (MobileNetV3/EfficientNet-Lite, see docs/MODEL_TRAINING.md §6). If your
/// fine-tune uses different preprocessing, update this alongside it — a
/// mismatch here silently degrades classification accuracy rather than
/// throwing.
- (nullable NSData *)preprocessClassifierInputWithPixelBuffer:(CVPixelBufferRef)pixelBuffer
                                                           roi:(CGRect)roi
                                                    targetSize:(int)targetSize
    NS_SWIFT_NAME(preprocessClassifierInput(withPixelBuffer:roi:targetSize:));

#pragma mark - Capture pipeline (file-based)

/// Loads an image file (JPEG/PNG via `cv::imread`, with a UIImage/ImageIO
/// fallback for formats like HEIC) into this bridge's internal "capture"
/// working image. Returns the loaded image's pixel size, or CGSizeZero if it
/// could not be read.
- (CGSize)loadCaptureImageAtPath:(NSString *)imagePath
    NS_SWIFT_NAME(loadCaptureImage(atPath:));

/// The current capture working image's pixel size (reflects perspective
/// rectification if `rectifyCaptureImage(with:outputPath:)` already ran).
- (CGSize)currentImageSize
    NS_SWIFT_NAME(currentImageSize());

/// Perspective-corrects the capture working image using `quad` (in the
/// loaded image's pixel space) via `cv::getPerspectiveTransform` +
/// `cv::warpPerspective`, sized to the quad's own width/height. Updates the
/// internal working image in place to the rectified result and writes it as
/// a JPEG to `outputPath`. Returns YES on success.
- (BOOL)rectifyCaptureImageWithQuad:(DSQuad *)quad
                          outputPath:(NSString *)outputPath
    NS_SWIFT_NAME(rectifyCaptureImage(with:outputPath:));

/// Resizes the current capture working image (keeping aspect ratio) so its
/// longer side is at most `maxSide`, rounded to a multiple of 32 (the
/// standard DBNet input constraint), and returns an ImageNet-normalised RGB
/// float32 NCHW tensor for the OCR detection model.
- (nullable NSData *)preprocessOcrDetInputWithMaxSide:(int)maxSide
                                              outWidth:(int *)outWidth
                                             outHeight:(int *)outHeight
    NS_SWIFT_NAME(preprocessOcrDetInput(withMaxSide:outWidth:outHeight:));

/// Post-processes a DBNet probability map (single-channel, `mapWidth` x
/// `mapHeight` float32) into rotated text-line boxes in the *current capture
/// image's* full pixel coordinate space, sorted top-to-bottom by vertical
/// center (reading order).
- (NSArray<DSTextBox *> *)textBoxesFromProbabilityMap:(NSData *)probabilityMap
                                              mapWidth:(int)mapWidth
                                             mapHeight:(int)mapHeight
                                             threshold:(double)threshold
    NS_SWIFT_NAME(textBoxes(fromProbabilityMap:mapWidth:mapHeight:threshold:));

/// Crops+deskews one text-line box out of the capture image (rotated-rect
/// aware, via `cv::getPerspectiveTransform`/`warpPerspective`, matching
/// RapidOCR's `get_rotate_crop_image`), optionally rotates it 180° (the cls
/// stage's correction), and resizes to a fixed height preserving aspect
/// ratio. Returns a `[-1, 1]`-normalised RGB float32 NCHW `[1,3,targetHeight,
/// outWidth]` tensor for the OCR recognition model.
- (nullable NSData *)preprocessRecInputForBox:(DSTextBox *)box
                                     rotate180:(BOOL)rotate180
                                  targetHeight:(int)targetHeight
                                      outWidth:(int *)outWidth
    NS_SWIFT_NAME(preprocessRecInput(for:rotate180:targetHeight:outWidth:));

/// Crops+deskews one text-line box (same crop as `preprocessRecInputForBox`)
/// into a fixed `clsWidth x clsHeight` canvas (aspect-preserved, zero-padded)
/// for the 2-class orientation classifier. Returns a `[-1, 1]`-normalised RGB
/// float32 NCHW tensor.
- (nullable NSData *)preprocessClsInputForBox:(DSTextBox *)box
                                      clsWidth:(int)clsWidth
                                     clsHeight:(int)clsHeight
    NS_SWIFT_NAME(preprocessClsInput(for:clsWidth:clsHeight:));

@end

NS_ASSUME_NONNULL_END
