//
//  ModelPathResolver.swift
//  react-native-doc-scanner
//
//  Resolves the `bundle://<filename>` path convention used by
//  `src/models/modelRegistry.ts` (see `ModelConfig`'s doc comment there) into a
//  real filesystem path `ONNXInference` can hand to `ORTSession`.
//

import Foundation
import NitroModules

enum ModelPathResolver {
  private static let bundleScheme = "bundle://"

  /// Resolves a `bundle://<name>` path (shipped inside this package's
  /// `assets/models/`, exposed via the podspec's `s.resources`) or an absolute
  /// filesystem path (anything not using the `bundle://` scheme — e.g. a model
  /// an app downloaded at runtime into `Library/` and registered via
  /// `ModelManager.register`) into a real path on disk.
  ///
  /// Throws `RuntimeError` if a `bundle://` resource can't be located in the
  /// app's main bundle — this is always a packaging bug (missing from
  /// `s.resources`, or the file was never copied into `assets/models/`, see
  /// docs/MODEL_TRAINING.md), so failing loudly here is correct rather than
  /// silently continuing with a session that will fail to load moments later
  /// with a much less clear error.
  static func resolve(_ path: String) throws -> String {
    guard path.hasPrefix(bundleScheme) else {
      // Not a bundle:// path — treat as an absolute filesystem path as-is.
      return path
    }

    let name = String(path.dropFirst(bundleScheme.count))
    guard !name.isEmpty else {
      throw RuntimeError.error(withMessage: "react-native-doc-scanner: \"\(path)\" is not a valid bundle:// model path (empty filename).")
    }

    let nsName = name as NSString
    let ext = nsName.pathExtension
    let base = ext.isEmpty ? name : nsName.deletingPathExtension

    if let resourcePath = Bundle.main.path(forResource: base, ofType: ext.isEmpty ? nil : ext) {
      return resourcePath
    }

    // Fall back to searching with the full filename as-is, in case it contains
    // multiple dots that aren't a simple "name.ext" split (e.g. a versioned
    // filename like "en_dict.v2.txt" where naive splitting would look for
    // resource "en_dict.v2" with extension "txt" under a *different* bundle
    // subdirectory setup than expected).
    if let resourcePath = Bundle.main.path(forResource: name, ofType: nil) {
      return resourcePath
    }

    throw RuntimeError.error(withMessage: "react-native-doc-scanner: could not find bundled model resource \"\(name)\" in the main bundle. Confirm it is listed under `s.resources` in react-native-doc-scanner.podspec and was actually copied into assets/models/ (see docs/MODEL_TRAINING.md and scripts/fetch-models.sh).")
  }
}
