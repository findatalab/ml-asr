import io
import torch
import torchaudio
import torchaudio.transforms as T
from flask import Flask, request, render_template, send_file

# Импортируем твою модель из соседнего файла
from model import BSRNN

app = Flask(__name__)

# Конфигурация
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SR = 16000
N_FFT = 512
HOP_LENGTH = 256
MODEL_PATH = 'models/bsrnn_weights_final.pth'

# Инициализация и загрузка модели при старте сервера
print("Загрузка модели...")
model = BSRNN(freq_bins=257, feature_dim=128, hidden_size=256, num_layers=2).to(DEVICE)
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.eval()
print("Модель готова к работе.")

@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')

@app.route('/process', methods=['POST'])
def process_audio():
    if 'audio' not in request.files:
        return "Файл не найден", 400

    file = request.files['audio']
    if file.filename == '':
        return "Файл не выбран", 400

    try:
        # Загружаем аудио напрямую из полученного файла
        waveform, sr = torchaudio.load(file)

        # Предобработка: ресемплинг и конвертация в моно
        if sr != SR:
            waveform = T.Resample(sr, SR)(waveform)
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True) # (1, T)

        waveform = waveform.to(DEVICE)
        
        # Добавляем размерность батча (1, 1, T)
        waveform = waveform.unsqueeze(0)

        # Инференс модели
        with torch.no_grad():
            X_comp = torch.stft(waveform.squeeze(1), n_fft=N_FFT, hop_length=HOP_LENGTH, return_complex=False)
            S_hat = model(X_comp)
            
            # Обратное преобразование Фурье
            denoised_tensor = torch.istft(
                torch.view_as_complex(S_hat), 
                n_fft=N_FFT, 
                hop_length=HOP_LENGTH, 
                length=waveform.shape[-1]
            )

        # Сохраняем очищенное аудио в буфер памяти
        out_buffer = io.BytesIO()
        # Переводим тензор обратно на CPU (1, T)
        output_waveform = denoised_tensor.squeeze(0).cpu()
        torchaudio.save(out_buffer, output_waveform, SR, format="wav")
        out_buffer.seek(0)

        return send_file(out_buffer, mimetype="audio/wav")

    except Exception as e:
        return str(e), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)