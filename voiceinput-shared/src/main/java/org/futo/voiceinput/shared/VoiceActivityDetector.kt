package org.futo.voiceinput.shared

import android.content.Context
import android.os.Build
import android.util.Log
import com.konovalov.vad.Vad
import com.konovalov.vad.config.FrameSize
import com.konovalov.vad.config.Mode
import com.konovalov.vad.config.Model
import com.konovalov.vad.config.SampleRate
import com.konovalov.vad.models.VadModel
import com.voxrt.silero.VadEvent
import com.voxrt.silero.VoxrtSileroVadEngine
import java.io.Closeable
import kotlin.math.pow
import kotlin.math.sqrt

/**
 * Streaming voice activity detector for 16 kHz mono PCM16 audio.
 *
 * Not thread-safe: all calls must be serialized by the caller.
 */
internal interface VoiceActivityDetector : Closeable {
    /** Feeds the first [count] samples of [samples] into the detector. */
    fun process(samples: ShortArray, count: Int)

    /** True if speech has been detected at any point since creation or the last [reset]. */
    val hasDetectedSpeech: Boolean

    /** Milliseconds of continuous non-speech up to the latest processed sample (0 while speaking). */
    val silenceMs: Int

    /** Clears all speech state (e.g. after the start sound has played). */
    fun reset()

    companion object {
        private const val TAG = "VoiceActivityDetector"

        /**
         * Creates the Silero (VoxRT) detector where supported, otherwise falls back to WebRTC VAD.
         * VoxRT requires API 26+ and only ships arm64-v8a and x86_64 native libraries.
         */
        fun create(context: Context): VoiceActivityDetector {
            if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.O) {
                try {
                    return SileroVoiceActivityDetector(context)
                } catch (e: Throwable) {
                    // UnsatisfiedLinkError / NoClassDefFoundError on unsupported ABIs,
                    // IllegalStateException if the model can't be loaded
                    Log.w(TAG, "Silero VAD unavailable, falling back to WebRTC VAD", e)
                }
            }
            return WebRtcVoiceActivityDetector()
        }
    }
}

private class SileroVoiceActivityDetector(context: Context) : VoiceActivityDetector {
    private val engine: VoxrtSileroVadEngine = context.assets.openFd(MODEL_ASSET).use { fd ->
        // The native side copies the model, so the fd can be closed immediately
        VoxrtSileroVadEngine.fromAssetFd(fd, VoxrtSileroVadEngine.Config(minSilenceMs = MIN_SILENCE_MS))
    }

    private var processedSamples = 0L
    private var isSpeaking = false
    // Time (relative to the engine clock) at which the current silence began
    private var silenceStartMs = 0L

    override var hasDetectedSpeech = false
        private set

    private val processedMs: Long
        get() = processedSamples * 1000L / SAMPLE_RATE

    override val silenceMs: Int
        get() = if (isSpeaking) 0 else (processedMs - silenceStartMs).coerceAtLeast(0L).toInt()

    override fun process(samples: ShortArray, count: Int) {
        if (count <= 0) return
        val pcm = if (count == samples.size) samples else samples.copyOf(count)

        for (event in engine.processPcm(pcm)) {
            when (event) {
                is VadEvent.SpeechOnset -> {
                    isSpeaking = true
                    hasDetectedSpeech = true
                }
                is VadEvent.SpeechOffset -> {
                    isSpeaking = false
                    // The offset is only reported after MIN_SILENCE_MS of silence
                    silenceStartMs = event.timeMs - MIN_SILENCE_MS
                }
            }
        }
        processedSamples += count
    }

    override fun reset() {
        engine.reset()
        processedSamples = 0L
        isSpeaking = false
        silenceStartMs = 0L
        hasDetectedSpeech = false
    }

    override fun close() {
        engine.close()
    }

    private companion object {
        const val MODEL_ASSET = "silero_vad.vxrt"
        const val SAMPLE_RATE = 16000
        // Short hangover for the engine's own hysteresis; the user-configured auto-stop
        // duration is applied on top of this via silenceMs
        const val MIN_SILENCE_MS = 100
    }
}

/**
 * Legacy WebRTC GMM detector, augmented with an adaptive energy check because on some
 * devices (e.g. Nothing Phone) the VOICE_RECOGNITION source has very low gain, so speech
 * never exceeds WebRTC VAD's internal energy floor.
 */
private class WebRtcVoiceActivityDetector : VoiceActivityDetector {
    private val vad: VadModel = Vad.builder().setModel(Model.WEB_RTC_GMM).setMode(Mode.VERY_AGGRESSIVE)
        .setFrameSize(FrameSize.FRAME_SIZE_480).setSampleRate(SampleRate.SAMPLE_RATE_16K)
        .setSpeechDurationMs(150).setSilenceDurationMs(300).build()

    private val frame = ShortArray(FRAME_SIZE)
    private var framePos = 0

    private var noiseFloor = Float.POSITIVE_INFINITY
    private var numConsecutiveNonSpeech = 0
    private var numConsecutiveSpeech = 0

    override var hasDetectedSpeech = false
        private set

    override val silenceMs: Int
        get() = numConsecutiveNonSpeech * FRAME_MS

    override fun process(samples: ShortArray, count: Int) {
        for (i in 0 until count) {
            frame[framePos++] = samples[i]
            if (framePos == FRAME_SIZE) {
                processFrame()
                framePos = 0
            }
        }
    }

    private fun processFrame() {
        val frameRms = sqrt(frame.sumOf { (it.toFloat() / Short.MAX_VALUE.toFloat()).pow(2).toDouble() } / frame.size).toFloat()
        noiseFloor = if (frameRms < noiseFloor) frameRms else noiseFloor * NOISE_FLOOR_RISE_PER_FRAME
        val speechThreshold = maxOf(noiseFloor * SPEECH_ABOVE_NOISE_RATIO, MIN_SPEECH_RMS)

        val isSpeech = vad.isSpeech(frame) || frameRms > speechThreshold
        if (isSpeech) {
            numConsecutiveNonSpeech = 0
            numConsecutiveSpeech++
            if (numConsecutiveSpeech > MIN_SPEECH_FRAMES) hasDetectedSpeech = true
        } else {
            numConsecutiveNonSpeech++
            numConsecutiveSpeech = 0
        }
    }

    override fun reset() {
        // Noise floor is intentionally kept, it remains valid
        numConsecutiveNonSpeech = 0
        numConsecutiveSpeech = 0
        hasDetectedSpeech = false
    }

    override fun close() {
        vad.close()
    }

    private companion object {
        // 480 samples at 16kHz = 30ms
        const val FRAME_SIZE = 480
        const val FRAME_MS = 30
        const val MIN_SPEECH_FRAMES = 8
        // ~18 dB above the measured background noise counts as speech
        const val SPEECH_ABOVE_NOISE_RATIO = 8.0f
        // Never consider anything below -60 dBFS speech
        const val MIN_SPEECH_RMS = 0.001f
        // Let the noise floor creep up (~1.7%/s) so it can follow increasing background noise
        const val NOISE_FLOOR_RISE_PER_FRAME = 1.0005f
    }
}
