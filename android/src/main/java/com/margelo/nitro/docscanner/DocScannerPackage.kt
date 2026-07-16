package com.margelo.nitro.docscanner

import com.facebook.react.BaseReactPackage
import com.facebook.react.bridge.NativeModule
import com.facebook.react.bridge.ReactApplicationContext
import com.facebook.react.module.model.ReactModuleInfoProvider

/**
 * A vestigial `ReactPackage` for `react-native-doc-scanner`.
 *
 * This module's actual Nitro HybridObject (`HybridDocScanner`) is registered natively via
 * JNI (`registerAllNatives()` in `DocScannerOnLoad.cpp`/`.hpp`, wired through
 * `nitrogen/generated/android/DocScanner+autolinking.{gradle,cmake}`) — it does NOT go through
 * the classic `NativeModule`/`ReactPackage` bridge at all.
 *
 * This class exists only so `@react-native-community/cli-config-android`'s dependency
 * autolinking (`findPackageClassName`, which regex-scans for a `class ... implements
 * ReactPackage`) actually discovers this package's `android/` sourceDir — without a class
 * like this one, `dependencyConfig()` returns `null` for the whole `android` platform entry
 * (see `@react-native-community/cli-config-android/build/config/index.js`), and Gradle's
 * `autolinkLibrariesFromCommand()` never `include()`s this module's Gradle project at all, so
 * none of the Kotlin/C++ in this package would even get compiled into the app. Every other
 * Nitro module in this workspace (`react-native-mmkv`'s `NitroMmkvPackage`, `react-native-vision-camera`'s
 * `VisionCameraPackage`, `react-native-nitro-image`'s `NitroImagePackage`) ships the same kind
 * of empty package for the same reason.
 *
 * The `init` block also ensures the native library is loaded as soon as RN instantiates this
 * package (relevant for Old Architecture bridge apps; New Architecture apps load it via
 * `HybridDocScanner`'s own construction, but calling this is idempotent either way).
 */
class DocScannerPackage : BaseReactPackage() {
    override fun getModule(name: String, reactContext: ReactApplicationContext): NativeModule? = null

    override fun getReactModuleInfoProvider(): ReactModuleInfoProvider = ReactModuleInfoProvider { HashMap() }

    companion object {
        init {
            DocScannerOnLoad.initializeNative()
        }
    }
}
