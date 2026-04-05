/*
 * mq135_reader.ino — MQ135 Air Quality Sensor Reader
 * For BREATHE / BioAQI project
 *
 * Wiring:
 *   MQ135 VCC  → Arduino 5V
 *   MQ135 GND  → Arduino GND
 *   MQ135 AOUT → Arduino A0
 *   (DOUT pin is not used)
 *
 * Output (Serial @ 9600 baud):
 *   PPM:450.23        (one line every 2 seconds)
 *
 * How it works:
 *   1. On startup: samples 50 readings to calibrate Ro (clean-air baseline).
 *      Run this FIRST TIME in clean outdoor air or a well-ventilated room.
 *   2. In loop: reads analog value → converts to Rs (sensor resistance)
 *      → computes ratio Rs/Ro → applies datasheet power-law curve → PPM.
 *   3. Sends "PPM:<value>" over Serial for Python (sensor.py) to parse.
 *
 * Calibration note:
 *   After the first calibration run, read the Ro value from Serial Monitor
 *   and hardcode it as RO_HARDCODED below to skip re-calibration on reboot.
 */

// ── Pin & communication ────────────────────────────────────────────────────
#define MQ135_PIN        A0
#define BAUD_RATE        9600

// ── Sensor constants ───────────────────────────────────────────────────────
// Load resistor on the MQ135 breakout board (typically 10 kΩ)
const float RLOAD = 10.0;

// Clean-air Rs/Ro ratio from MQ135 datasheet
const float RATIO_CLEAN_AIR = 3.6;

// Number of samples to average during startup calibration
#define CALIBRATION_SAMPLES  50
#define CALIBRATION_DELAY_MS 100   // 100 ms between samples → ~5 s total

// ── PPM formula constants (CO₂-proxy curve from MQ135 datasheet) ──────────
// PPM = CURVE_A * (Rs/Ro) ^ CURVE_B
const float CURVE_A = 116.6020682;
const float CURVE_B = -2.769034857;

// ── Optional: hardcode Ro after first calibration run ─────────────────────
// Set to 0.0 to always auto-calibrate on startup.
// Example: after seeing "Calibrated Ro = 9.83 kΩ" in Serial Monitor,
//          set this to 9.83 to skip re-calibration.
const float RO_HARDCODED = 0.0;

// ── Runtime state ─────────────────────────────────────────────────────────
float Ro = 20.0;   // Will be updated by calibration

// ── Helpers ────────────────────────────────────────────────────────────────

float read_rs() {
  int raw = analogRead(MQ135_PIN);
  if (raw == 0) raw = 1;   // avoid division by zero
  return ((1023.0 / (float)raw) - 1.0) * RLOAD;
}

float rs_to_ppm(float rs) {
  float ratio = rs / Ro;
  return CURVE_A * pow(ratio, CURVE_B);
}

// ── Setup ──────────────────────────────────────────────────────────────────

void setup() {
  Serial.begin(BAUD_RATE);
  delay(500);   // let serial stabilise

  if (RO_HARDCODED > 0.0) {
    Ro = RO_HARDCODED;
    Serial.print("INFO:Using hardcoded Ro=");
    Serial.println(Ro, 2);
    return;
  }

  // Auto-calibrate Ro in clean air
  Serial.println("INFO:Calibrating... keep sensor in clean air");
  float sum = 0.0;
  for (int i = 0; i < CALIBRATION_SAMPLES; i++) {
    float rs = read_rs();
    sum += rs / RATIO_CLEAN_AIR;
    delay(CALIBRATION_DELAY_MS);
  }
  Ro = sum / CALIBRATION_SAMPLES;

  Serial.print("INFO:Calibrated Ro=");
  Serial.print(Ro, 2);
  Serial.println(" kohm — copy this value to RO_HARDCODED to skip next time");
  Serial.println("INFO:Starting readings...");
}

// ── Main loop ──────────────────────────────────────────────────────────────

void loop() {
  int   raw = analogRead(MQ135_PIN);
  float rs  = (raw == 0) ? RLOAD * 1023.0 : ((1023.0 / (float)raw) - 1.0) * RLOAD;
  float ppm = CURVE_A * pow(rs / Ro, CURVE_B);

  // Only cap the upper end; let low values pass through for debugging
  if (ppm < 0)      ppm = 0;
  if (ppm > 10000)  ppm = 10000;

  // Send parseable line: "PPM:450.23,RAW:712"
  // RAW helps diagnose calibration — in clean air expect RAW ~300-700
  Serial.print("PPM:");
  Serial.print(ppm, 1);
  Serial.print(",RAW:");
  Serial.println(raw);

  delay(1000);   // one reading per second for responsive updates
}
