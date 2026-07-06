#include "model_data.h"

#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/schema/schema_generated.h"
// =====================================================
// PINES DE LOS POTENCIOMETROS
// =====================================================

const int POT_RR    = 28;  // GP28
const int POT_PEAKS = 27;  // GP27
const int POT_FREQ  = 26;  // GP26


// =====================================================
// RANGOS DE LAS FEATURES
// Obtenidos a partir de X_train
// =====================================================

const float FEATURE_MIN[3] = {
  0.499f,
  3.0f,
  0.6f
};

const float FEATURE_MAX[3] = {
  1.524f,
  9.0f,
  6.0f
};


// =====================================================
// PARAMETROS DEL StandardScaler
// =====================================================

const float SCALER_MEAN[3] = {
  0.8880125f,
  6.33333333f,
  3.56333333f
};

const float SCALER_SCALE[3] = {
  0.43637098f,
  2.49443826f,
  1.84860007f
};

// =====================================================
// TENSORFLOW LITE MICRO
// =====================================================

const tflite::Model* tflite_model = nullptr;

tflite::MicroInterpreter* interpreter = nullptr;

TfLiteTensor* model_input = nullptr;

TfLiteTensor* model_output = nullptr;


// Memoria de trabajo del modelo
constexpr int TENSOR_ARENA_SIZE = 64 * 1024;

alignas(16) uint8_t tensor_arena[TENSOR_ARENA_SIZE];

// =====================================================
// FUNCIONES
// =====================================================

// Convierte ADC 0-4095 al rango real de la característica
float adcToFeature(
  int adc,
  float minValue,
  float maxValue
) {
  return minValue
       + ((float)adc / 4095.0f)
       * (maxValue - minValue);
}


// Estandarización:
// z = (x - media) / desviación
float standardize(
  float value,
  float mean,
  float scale
) {
  return (value - mean) / scale;
}


// =====================================================
// SETUP
// =====================================================

void setup() {

  Serial1.begin(115200);

  analogReadResolution(12);

  delay(1000);

  Serial1.println();
  Serial1.println("==================================");
  Serial1.println(" CLASIFICADOR DE ARRITMIAS");
  Serial1.println(" Raspberry Pi Pico");
  Serial1.println("==================================");


  // ==================================================
  // CARGAR MODELO
  // ==================================================

  tflite_model = tflite::GetModel(
    arrhythmia_model_tflite
  );


  // Verificar compatibilidad
  if (tflite_model->version() != TFLITE_SCHEMA_VERSION) {

    Serial1.println(
      "ERROR: modelo TFLite incompatible"
    );

    while (true) {
      delay(1000);
    }
  }


  // Registrar operaciones de la red
  static tflite::MicroMutableOpResolver<3> resolver;

  resolver.AddFullyConnected();
  resolver.AddRelu();
  resolver.AddSoftmax();


  // Crear intérprete
  static tflite::MicroInterpreter static_interpreter(
    tflite_model,
    resolver,
    tensor_arena,
    TENSOR_ARENA_SIZE
  );

  interpreter = &static_interpreter;


  // Reservar memoria para tensores
  if (interpreter->AllocateTensors() != kTfLiteOk) {

    Serial1.println(
      "ERROR: AllocateTensors fallo"
    );

    while (true) {
      delay(1000);
    }
  }


  // Obtener entrada y salida
  model_input = interpreter->input(0);

  model_output = interpreter->output(0);


  Serial1.println(
    "Modelo cargado correctamente"
  );
}


// =====================================================
// LOOP
// =====================================================

void loop() {

  // -----------------------------------------------
  // 1. Lectura ADC
  // -----------------------------------------------

  int adc_rr = analogRead(POT_RR);

  int adc_peaks = analogRead(POT_PEAKS);

  int adc_freq = analogRead(POT_FREQ);


  // -----------------------------------------------
  // 2. ADC -> características originales
  // -----------------------------------------------

  float mean_rr = adcToFeature(
    adc_rr,
    FEATURE_MIN[0],
    FEATURE_MAX[0]
  );


  float num_peaks = adcToFeature(
    adc_peaks,
    FEATURE_MIN[1],
    FEATURE_MAX[1]
  );


  float dominant_freq = adcToFeature(
    adc_freq,
    FEATURE_MIN[2],
    FEATURE_MAX[2]
  );


  // -----------------------------------------------
  // 3. Normalización igual que StandardScaler
  // -----------------------------------------------

  float input_scaled[3];


  input_scaled[0] = standardize(
    mean_rr,
    SCALER_MEAN[0],
    SCALER_SCALE[0]
  );


  input_scaled[1] = standardize(
    num_peaks,
    SCALER_MEAN[1],
    SCALER_SCALE[1]
  );


  input_scaled[2] = standardize(
    dominant_freq,
    SCALER_MEAN[2],
    SCALER_SCALE[2]
  );

  // -----------------------------------------------
// 4. COPIAR FEATURES AL TENSOR DE ENTRADA
// -----------------------------------------------

model_input->data.f[0] = input_scaled[0];

model_input->data.f[1] = input_scaled[1];

model_input->data.f[2] = input_scaled[2];


// -----------------------------------------------
// 5. EJECUTAR INFERENCIA
// -----------------------------------------------

if (interpreter->Invoke() != kTfLiteOk) {

  Serial1.println(
    "ERROR durante la inferencia"
  );

  delay(1000);

  return;
}


// -----------------------------------------------
// 6. LEER SALIDAS
// -----------------------------------------------

float prob_40 = model_output->data.f[0];

float prob_90 = model_output->data.f[1];

float prob_120 = model_output->data.f[2];


// Buscar la clase con mayor probabilidad

int predicted_class = 0;

float max_prob = prob_40;


if (prob_90 > max_prob) {

  max_prob = prob_90;

  predicted_class = 1;
}


if (prob_120 > max_prob) {

  max_prob = prob_120;

  predicted_class = 2;
}


  // -----------------------------------------------
  // 4. Mostrar valores
  // -----------------------------------------------

  Serial1.println();
  Serial1.println("----------------------------------");

  Serial1.println("VALORES ADC:");

  Serial1.print("GP28 = ");
  Serial1.println(adc_rr);

  Serial1.print("GP27 = ");
  Serial1.println(adc_peaks);

  Serial1.print("GP26 = ");
  Serial1.println(adc_freq);


  Serial1.println();

  Serial1.println("FEATURES ORIGINALES:");

  Serial1.print("mean_rr = ");
  Serial1.println(mean_rr, 4);

  Serial1.print("num_peaks = ");
  Serial1.println(num_peaks, 4);

  Serial1.print("dominant_freq = ");
  Serial1.println(dominant_freq, 4);


  Serial1.println();

  Serial1.println("ENTRADA NORMALIZADA DEL MODELO:");

  Serial1.print("x[0] = ");
  Serial1.println(input_scaled[0], 4);

  Serial1.print("x[1] = ");
  Serial1.println(input_scaled[1], 4);

  Serial1.print("x[2] = ");
  Serial1.println(input_scaled[2], 4);

Serial1.println();

Serial1.println("PROBABILIDADES:");

Serial1.print("40 bpm: ");
Serial1.println(prob_40, 4);

Serial1.print("90 bpm: ");
Serial1.println(prob_90, 4);

Serial1.print("120 bpm: ");
Serial1.println(prob_120, 4);


Serial1.println();

Serial1.print("PREDICCION: ");


if (predicted_class == 0) {

  Serial1.println("40 bpm");

}
else if (predicted_class == 1) {

  Serial1.println("90 bpm");

}
else {

  Serial1.println("120 bpm");

}

  delay(1000);
}