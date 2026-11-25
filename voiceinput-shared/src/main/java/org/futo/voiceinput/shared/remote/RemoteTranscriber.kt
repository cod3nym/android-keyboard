package org.futo.voiceinput.shared.remote

import kotlinx.serialization.json.Json
import kotlinx.serialization.json.jsonObject
import kotlinx.serialization.json.jsonPrimitive
import java.io.ByteArrayOutputStream
import java.io.DataOutputStream
import java.io.IOException
import java.net.HttpURLConnection
import java.net.URL
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.util.UUID

/**
 * Configuration for an OpenAI-compatible `/v1/audio/transcriptions` endpoint, such as a
 * self-hosted faster-whisper / speaches / LocalAI / whisper.cpp server.
 *
 * [baseUrl] may be either the API base (e.g. `http://192.168.1.10:8000/v1`) or the full
 * endpoint URL (ending in `/audio/transcriptions`).
 */
data class RemoteTranscriptionConfiguration(
    val baseUrl: String,
    val apiKey: String,
    val model: String,
    val connectTimeoutMs: Int = 10_000,
    val readTimeoutMs: Int = 120_000,
)

class RemoteTranscriptionException(message: String, cause: Throwable? = null) : Exception(message, cause)

private const val SAMPLE_RATE = 16000

fun RemoteTranscriptionConfiguration.endpointUrl(): String {
    val trimmed = baseUrl.trim().trimEnd('/')
    return when {
        trimmed.endsWith("/audio/transcriptions") -> trimmed
        else -> "$trimmed/audio/transcriptions"
    }
}

/** Encodes mono float samples in [-1, 1] at 16kHz as a 16-bit PCM WAV file. */
internal fun encodeWav(samples: FloatArray): ByteArray {
    val dataSize = samples.size * 2
    val buffer = ByteBuffer.allocate(44 + dataSize).order(ByteOrder.LITTLE_ENDIAN)

    buffer.put("RIFF".toByteArray(Charsets.US_ASCII))
    buffer.putInt(36 + dataSize)
    buffer.put("WAVE".toByteArray(Charsets.US_ASCII))

    buffer.put("fmt ".toByteArray(Charsets.US_ASCII))
    buffer.putInt(16)                 // PCM chunk size
    buffer.putShort(1)                // Audio format: PCM
    buffer.putShort(1)                // Channels: mono
    buffer.putInt(SAMPLE_RATE)
    buffer.putInt(SAMPLE_RATE * 2)    // Byte rate
    buffer.putShort(2)                // Block align
    buffer.putShort(16)               // Bits per sample

    buffer.put("data".toByteArray(Charsets.US_ASCII))
    buffer.putInt(dataSize)
    for (s in samples) {
        buffer.putShort((s.coerceIn(-1.0f, 1.0f) * Short.MAX_VALUE).toInt().toShort())
    }

    return buffer.array()
}

class RemoteTranscriber(private val config: RemoteTranscriptionConfiguration) {
    @Volatile
    private var activeConnection: HttpURLConnection? = null

    @Volatile
    private var cancelled = false

    fun cancel() {
        cancelled = true
        activeConnection?.disconnect()
    }

    /**
     * Blocking call; run it on an IO dispatcher.
     *
     * @param language ISO-639-1 code to force, or null to let the server auto-detect.
     * @param prompt Optional prompt to bias decoding (e.g. a glossary of personal words).
     */
    @Throws(RemoteTranscriptionException::class)
    fun transcribe(samples: FloatArray, language: String?, prompt: String?): String {
        cancelled = false

        val url = try {
            URL(config.endpointUrl())
        } catch (e: Exception) {
            throw RemoteTranscriptionException("Invalid server URL: ${config.baseUrl}", e)
        }

        val boundary = "----FUTOVoiceInput" + UUID.randomUUID().toString().replace("-", "")
        val body = buildMultipartBody(boundary, encodeWav(samples), language, prompt)

        val connection = try {
            url.openConnection() as HttpURLConnection
        } catch (e: IOException) {
            throw RemoteTranscriptionException("Could not connect to ${url.host}", e)
        }
        activeConnection = connection

        try {
            connection.requestMethod = "POST"
            connection.doOutput = true
            connection.useCaches = false
            connection.connectTimeout = config.connectTimeoutMs
            connection.readTimeout = config.readTimeoutMs
            connection.setFixedLengthStreamingMode(body.size)
            connection.setRequestProperty("Content-Type", "multipart/form-data; boundary=$boundary")
            connection.setRequestProperty("Accept", "application/json")
            if (config.apiKey.isNotBlank()) {
                connection.setRequestProperty("Authorization", "Bearer ${config.apiKey.trim()}")
            }

            connection.outputStream.use { it.write(body) }

            val code = connection.responseCode
            val responseText = (if (code in 200..299) connection.inputStream else connection.errorStream)
                ?.bufferedReader()?.use { it.readText() } ?: ""

            if (code !in 200..299) {
                throw RemoteTranscriptionException("Server returned HTTP $code: ${responseText.take(200)}")
            }

            return parseResponse(responseText)
        } catch (e: RemoteTranscriptionException) {
            throw e
        } catch (e: IOException) {
            if (cancelled) throw RemoteTranscriptionException("Cancelled", e)
            throw RemoteTranscriptionException(e.message ?: e.javaClass.simpleName, e)
        } finally {
            connection.disconnect()
            activeConnection = null
        }
    }

    private fun parseResponse(responseText: String): String {
        val trimmed = responseText.trim()
        if (!trimmed.startsWith("{")) {
            // Some servers ignore response_format and return plain text
            return trimmed
        }

        return try {
            Json.parseToJsonElement(trimmed).jsonObject["text"]?.jsonPrimitive?.content
                ?: throw RemoteTranscriptionException("Response did not contain a \"text\" field")
        } catch (e: RemoteTranscriptionException) {
            throw e
        } catch (e: Exception) {
            throw RemoteTranscriptionException("Could not parse server response", e)
        }
    }

    private fun buildMultipartBody(
        boundary: String,
        wav: ByteArray,
        language: String?,
        prompt: String?
    ): ByteArray {
        val out = ByteArrayOutputStream(wav.size + 1024)
        val writer = DataOutputStream(out)
        val lineEnd = "\r\n"

        fun field(name: String, value: String) {
            writer.writeBytes("--$boundary$lineEnd")
            writer.writeBytes("Content-Disposition: form-data; name=\"$name\"$lineEnd")
            writer.writeBytes("Content-Type: text/plain; charset=UTF-8$lineEnd$lineEnd")
            writer.write(value.toByteArray(Charsets.UTF_8))
            writer.writeBytes(lineEnd)
        }

        writer.writeBytes("--$boundary$lineEnd")
        writer.writeBytes("Content-Disposition: form-data; name=\"file\"; filename=\"audio.wav\"$lineEnd")
        writer.writeBytes("Content-Type: audio/wav$lineEnd$lineEnd")
        writer.write(wav)
        writer.writeBytes(lineEnd)

        field("model", config.model.ifBlank { "whisper-1" })
        field("response_format", "json")
        if (!language.isNullOrBlank()) field("language", language)
        if (!prompt.isNullOrBlank()) field("prompt", prompt)

        writer.writeBytes("--$boundary--$lineEnd")
        writer.flush()

        return out.toByteArray()
    }
}
