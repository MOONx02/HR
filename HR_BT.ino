#include <Wire.h>
#include "MAX30105.h"            // SparkFun MAX3010x Pulse & Proximity Sensor Library
#include "spo2_algorithm.h"      // make sure spo2_algorithm.h/.cpp are in the same folder
#include "BluetoothSerial.h"     // ESP32 Bluetooth Library

MAX30105 particle;
BluetoothSerial SerialBT;

// IMPORTANT: Set a unique device name for each ESP32
// Change this to "ESP32_HR_1", "ESP32_HR_2", "ESP32_HR_3" for each device
#define DEVICE_NAME "ESP32_HR_3"
#define DEVICE_ID 3  // Change this to 1, 2, or 3 for each device

#define WINDOW_SIZE 100  // number of samples per second

uint32_t irBuffer[WINDOW_SIZE];
uint32_t redBuffer[WINDOW_SIZE];

// Store historical heart rate data for metrics calculation
#define HISTORY_SIZE 60  // Store last 60 readings (1 minute if reading every second)
int heartRateHistory[HISTORY_SIZE];
int historyIndex = 0;
int historyCount = 0;

void setup() {
  Serial.begin(115200);
  delay(1000);
  Serial.println("\n=== MAX30102 Heart Rate & SpO₂ Monitor with Bluetooth ===");
  Serial.println("Initializing sensor...\n");

  // Initialize Bluetooth
  if (!SerialBT.begin(DEVICE_NAME)) {
    Serial.println("❌ Bluetooth initialization failed!");
    while (1);
  }
  Serial.print("✅ Bluetooth initialized: ");
  Serial.println(DEVICE_NAME);
  Serial.println("Waiting for connection...");

  Wire.begin(21, 22);  // ESP32 default I2C pins

  if (!particle.begin(Wire)) {
    Serial.println("❌ Sensor not found. Check wiring and power.");
    while (1);
  }

  // Configure sensor parameters
  byte ledBrightness = 60;    // 0–255
  byte sampleAverage = 4;
  byte ledMode = 2;           // 2 = Red + IR
  int sampleRate = 100;       // Hz
  int pulseWidth = 411;       // µs
  int adcRange = 16384;       // nA

  particle.setup(ledBrightness, sampleAverage, ledMode,
                 sampleRate, pulseWidth, adcRange);

  Serial.println("✅ Sensor initialized successfully!");
  Serial.println("Place your finger gently on the sensor.\n");
  Serial.println("---------------------------------------------------");
}

void addToHistory(int heartRate) {
  // Only add valid heart rates (reasonable range)
  if (heartRate >= 40 && heartRate <= 200) {
    heartRateHistory[historyIndex] = heartRate;
    historyIndex = (historyIndex + 1) % HISTORY_SIZE;
    if (historyCount < HISTORY_SIZE) {
      historyCount++;
    }
  }
}

void loop() {
  static uint32_t lastCalc = 0;
  static int idx = 0;

  particle.check();

  while (particle.available()) {
    irBuffer[idx]  = particle.getFIFOIR();
    redBuffer[idx] = particle.getFIFORed();
    idx++;
    particle.nextSample();
  }

  // Compute once per second
  if (idx >= WINDOW_SIZE && millis() - lastCalc >= 1000) {
    lastCalc = millis();

    // Calculate average IR value to check finger contact quality
    uint32_t irSum = 0;
    uint32_t irMin = UINT32_MAX;
    uint32_t irMax = 0;
    for (int i = 0; i < WINDOW_SIZE; i++) {
      irSum += irBuffer[i];
      if (irBuffer[i] < irMin) irMin = irBuffer[i];
      if (irBuffer[i] > irMax) irMax = irBuffer[i];
    }
    uint32_t irAvg = irSum / WINDOW_SIZE;
    uint32_t irRange = irMax - irMin;

    // Print IR stats for debugging (only to Serial, not Bluetooth)
    Serial.print("📊 IR Avg: ");
    Serial.print(irAvg);
    Serial.print("   |   Min: ");
    Serial.print(irMin);
    Serial.print("   |   Max: ");
    Serial.print(irMax);
    Serial.print("   |   Range: ");
    Serial.println(irRange);

    // Check for good sensor contact
    const uint32_t IR_THRESHOLD = 50000;
    const uint32_t RANGE_MAX = 250;
    
    if (irAvg < IR_THRESHOLD) {
      Serial.println("⚠️  **POOR SENSOR READING** - Check finger placement");
      
      // Send status to Bluetooth
      if (SerialBT.hasClient()) {
        SerialBT.print("DEV:");
        SerialBT.print(DEVICE_ID);
        SerialBT.print(",STATUS:NO_FINGER");
        SerialBT.println();
      }
    } else {
      int32_t spo2 = 0, heartRate = 0;
      int8_t spo2Valid = 0, hrValid = 0;

      maxim_heart_rate_and_oxygen_saturation(
          irBuffer, WINDOW_SIZE,
          redBuffer,
          &spo2, &spo2Valid,
          &heartRate, &hrValid);

      // Print to Serial Monitor
      Serial.print("❤️ Heart Rate: ");
      if (hrValid)
        Serial.print(heartRate);
      else
        Serial.print("--");
      Serial.print(" bpm");

      Serial.print("   |   🩸 SpO₂: ");
      if (spo2Valid)
        Serial.print(spo2);
      else
        Serial.print("--");
      Serial.print("%");
      
      if (irRange > RANGE_MAX) {
        Serial.print("   *degraded*");
      }
      
      Serial.println();

      // Send data via Bluetooth in CSV format
      // Format: DEV:id,HR:value,HR_VALID:1/0,SPO2:value,SPO2_VALID:1/0,TIMESTAMP:millis
      if (SerialBT.hasClient()) {
        SerialBT.print("DEV:");
        SerialBT.print(DEVICE_ID);
        SerialBT.print(",HR:");
        SerialBT.print(heartRate);
        SerialBT.print(",HR_VALID:");
        SerialBT.print(hrValid);
        SerialBT.print(",SPO2:");
        SerialBT.print(spo2);
        SerialBT.print(",SPO2_VALID:");
        SerialBT.print(spo2Valid);
        SerialBT.print(",IR_AVG:");
        SerialBT.print(irAvg);
        SerialBT.print(",IR_RANGE:");
        SerialBT.print(irRange);
        SerialBT.print(",TIMESTAMP:");
        SerialBT.print(millis());
        SerialBT.println();
      } else {
        Serial.println("📡 No Bluetooth client connected");
      }

      // Add to history if heart rate is valid
      if (hrValid) {
        addToHistory(heartRate);
      }
    }

    Serial.println("---------------------------------------------------");

    idx = 0;  // reset buffer for next second
  }
}

