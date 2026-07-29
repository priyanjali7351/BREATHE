/*
 * gp2y10_reader.ino — Sharp GP2Y1010AU0F optical dust sensor reader
 * For BREATHE / BioAQI project
 *
 * Measures particulate matter (approx PM2.5) via IR light scattering and
 * sends it over serial for the PC-side reader (gp2y10.py) to parse.
 *
 * Wiring (see project docs):
 *   5V ──[220Ω]──┬── Pin 1 (V-LED)
 *                └──[220–470µF cap, − leg]── GND
 *   Pin 2 (LED-GND) → GND
 *   Pin 3 (LED)     → D2   (LED_PIN below)
 *   Pin 4 (S-GND)   → GND
 *   Pin 5 (Vo)      → A0   (MEASURE_PIN below)
 *   Pin 6 (Vcc)     → 5V
 *
 * Output (Serial @ 9600 baud), one line per second:
 *   PM25:41.2,RAW:183
 *   (PM25 = approx µg/m³, RAW = median 10-bit ADC of the pulse)
 *   INFO:... lines during startup calibration (ignored by the PC reader)
 *
 * How it works:
 *   1. Pulse the internal IR LED (Chris Nafis timing): LED on, wait 280µs,
 *      sample Vo, wait 40µs, LED off, idle to complete a 10ms cycle.
 *   2. Take the MEDIAN of N pulses to fight the sensor's inherent noise.
 *   3. On startup, sample clean air to learn the no-dust baseline voltage
 *      (Vclean) — same idea as the MQ135 sketch's Ro calibration.
 *   4. Convert: µg/m³ = (V − Vclean) / 0.005   (datasheet sensitivity
 *      0.5V per 0.1 mg/m³ → 0.005 V per µg/m³).
 *
 * IMPORTANT — LED polarity:
 *   This uses the canonical LOW = LED-on convention. If readings look
 *   inverted or pinned at zero, flip LED_ON / LED_OFF below.
 */

// ── Pins & comms ────────────────────────────────────────────────────────────
#define LED_PIN      2      // Pin 3 of the sensor (LED pulse control)
#define MEASURE_PIN  A0     // Pin 5 of the sensor (Vo analog output)
#define BAUD_RATE    9600

#define LED_ON       LOW    // flip to HIGH if readings look wrong
#define LED_OFF      HIGH

// ── Pulse timing (microseconds) — do NOT change ─────────────────────────────
const unsigned int SAMPLING_TIME = 280;    // LED on → wait before sampling Vo
const unsigned int DELTA_TIME     = 40;    // after sampling → before LED off
const unsigned int SLEEP_TIME     = 9680;  // idle → total cycle = 10ms

// ── Smoothing & rate ────────────────────────────────────────────────────────
#define SAMPLES_PER_READING  15    // median window (15 × 10ms ≈ 150ms)
#define ADC_REF_VOLTS        5.0
#define ADC_MAX              1024.0

// Datasheet sensitivity: 0.5V per 0.1 mg/m³ = 0.005 V per µg/m³.
const float VOLTS_PER_UGM3 = 0.005;

// ── Optional: hardcode the clean-air baseline after one calibration run ──────
// Set to 0.0 to auto-calibrate on every startup. After seeing
// "INFO:Calibrated Vclean=0.62" you may hardcode it here to skip calibration.
const float VCLEAN_HARDCODED = 0.0;

// ── Runtime state ───────────────────────────────────────────────────────────
float Vclean = 0.60;   // no-dust baseline voltage (updated by calibration)


// ── One LED pulse → raw ADC ─────────────────────────────────────────────────
int readPulseADC() {
  digitalWrite(LED_PIN, LED_ON);
  delayMicroseconds(SAMPLING_TIME);
  int adc = analogRead(MEASURE_PIN);
  delayMicroseconds(DELTA_TIME);
  digitalWrite(LED_PIN, LED_OFF);
  delayMicroseconds(SLEEP_TIME);
  return adc;
}

// ── Median of N pulses (robust against spikes) ──────────────────────────────
int readMedianADC() {
  int buf[SAMPLES_PER_READING];
  for (int i = 0; i < SAMPLES_PER_READING; i++) {
    buf[i] = readPulseADC();
  }
  // Simple insertion sort (N is small)
  for (int i = 1; i < SAMPLES_PER_READING; i++) {
    int key = buf[i], j = i - 1;
    while (j >= 0 && buf[j] > key) { buf[j + 1] = buf[j]; j--; }
    buf[j + 1] = key;
  }
  return buf[SAMPLES_PER_READING / 2];
}

float adcToVolts(int adc) {
  return adc * (ADC_REF_VOLTS / ADC_MAX);
}


// ── Setup ───────────────────────────────────────────────────────────────────
void setup() {
  pinMode(LED_PIN, OUTPUT);
  digitalWrite(LED_PIN, LED_OFF);
  Serial.begin(BAUD_RATE);
  delay(500);

  if (VCLEAN_HARDCODED > 0.0) {
    Vclean = VCLEAN_HARDCODED;
    Serial.print("INFO:Using hardcoded Vclean=");
    Serial.println(Vclean, 3);
    return;
  }

  // Auto-calibrate the no-dust baseline in clean air (~3s of medians)
  Serial.println("INFO:Calibrating baseline - keep sensor in clean air");
  float sum = 0.0;
  const int N = 20;
  for (int i = 0; i < N; i++) {
    sum += adcToVolts(readMedianADC());
  }
  Vclean = sum / N;
  Serial.print("INFO:Calibrated Vclean=");
  Serial.print(Vclean, 3);
  Serial.println(" V - copy to VCLEAN_HARDCODED to skip next time");
  Serial.println("INFO:Starting readings");
}


// ── Main loop ───────────────────────────────────────────────────────────────
void loop() {
  int   adc     = readMedianADC();
  float volts   = adcToVolts(adc);
  float ugm3    = (volts - Vclean) / VOLTS_PER_UGM3;
  if (ugm3 < 0) ugm3 = 0;          // clean air can dip below baseline noise
  if (ugm3 > 1000) ugm3 = 1000;    // clamp to sane range

  Serial.print("PM25:");
  Serial.print(ugm3, 1);
  Serial.print(",RAW:");
  Serial.println(adc);

  delay(850);   // ~1 reading/sec (medians already took ~150ms)
}
