import os
import numpy as np
import soundfile as sf
import librosa

def detect_silence(audio_data, sr, threshold_db=-60, min_silence_duration=0.1):
    """
    Phát hiện các đoạn silence ở đầu và cuối file audio
    
    Parameters:
    - audio_data: numpy array chứa dữ liệu audio
    - sr: sample rate
    - threshold_db: ngưỡng âm lượng để xác định silence (dB)
    - min_silence_duration: độ dài tối thiểu của đoạn silence (giây)
    
    Returns:
    - start_trim: số samples cần cắt ở đầu
    - end_trim: số samples cần cắt ở cuối
    """
    
    # Nếu audio là stereo, chuyển thành mono bằng cách lấy trung bình
    if len(audio_data.shape) > 1:
        audio_data = np.mean(audio_data, axis=1)
    
    # Chuyển sang dB
    magnitude = np.abs(audio_data)
    max_magnitude = np.max(magnitude)
    
    # Tránh chia cho 0 bằng cách thêm một số rất nhỏ
    eps = np.finfo(float).eps
    db = 20 * np.log10(magnitude / (max_magnitude + eps) + eps)
    
    # Tìm các điểm silence
    silence_mask = db < threshold_db
    
    # Tìm silence ở đầu
    start_trim = 0
    for i in range(len(silence_mask)):
        if not silence_mask[i]:
            if i > min_silence_duration * sr:
                start_trim = int(i - min_silence_duration * sr)
            break
    
    # Tìm silence ở cuối
    end_trim = len(audio_data)
    for i in range(len(silence_mask)-1, -1, -1):
        if not silence_mask[i]:
            if len(audio_data) - i > min_silence_duration * sr:
                end_trim = int(i + min_silence_duration * sr)
            break
    
    return start_trim, end_trim

def apply_fade(audio_data, sr, fade_duration=0.25):
    """
    Áp dụng fade in/out
    
    Parameters:
    - audio_data: numpy array chứa dữ liệu audio
    - sr: sample rate
    - fade_duration: độ dài fade (giây)
    
    Returns:
    - audio_data: dữ liệu audio sau khi áp dụng fade
    """
    fade_length = int(fade_duration * sr)
    
    # Tạo fade curve
    fade_in = np.linspace(0, 1, fade_length)
    fade_out = np.linspace(1, 0, fade_length)
    
    # Xử lý cho cả stereo và mono
    if len(audio_data.shape) > 1:
        # Stereo
        audio_data[:fade_length, :] *= fade_in[:, np.newaxis]
        audio_data[-fade_length:, :] *= fade_out[:, np.newaxis]
    else:
        # Mono
        audio_data[:fade_length] *= fade_in
        audio_data[-fade_length:] *= fade_out
    
    return audio_data

def process_wav_file(input_path, output_path, max_silence_duration=0.5):
    """
    Xử lý file WAV: cắt silence và áp dụng fade
    
    Parameters:
    - input_path: đường dẫn file input
    - output_path: đường dẫn file output
    - max_silence_duration: độ dài tối đa của silence (giây)
    """
    
    # Đọc file audio
    audio_data, sr = sf.read(input_path)
    
    # Lưu lại số kênh (mono/stereo)
    is_stereo = len(audio_data.shape) > 1
    
    # Phát hiện silence (sử dụng bản sao để tránh thay đổi dữ liệu gốc)
    start_trim, end_trim = detect_silence(audio_data.copy(), sr)
    
    # Cắt audio giữ lại silence tối đa max_silence_duration
    trimmed_audio = audio_data[start_trim:end_trim]
    
    # Áp dụng fade
    processed_audio = apply_fade(trimmed_audio, sr)
    
    # Lưu file với định dạng giống hệt file gốc
    sf.write(output_path, processed_audio, sr, subtype=sf.info(input_path).subtype)

def process_folder(input_folder, output_folder):
    """
    Xử lý tất cả file WAV trong folder
    
    Parameters:
    - input_folder: đường dẫn folder chứa file input
    - output_folder: đường dẫn folder chứa file output
    """
    
    # Tạo output folder nếu chưa tồn tại
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    # Xử lý từng file
    for filename in os.listdir(input_folder):
        if filename.lower().endswith('.wav'):
            input_path = os.path.join(input_folder, filename)
            output_path = os.path.join(output_folder, filename)
            
            try:
                process_wav_file(input_path, output_path)
                print(f"Đã xử lý thành công: {filename}")
            except Exception as e:
                print(f"Lỗi khi xử lý file {filename}: {str(e)}")

if __name__ == "__main__":
    # Nhập đường dẫn folder input và output
    print("Nhập đường đẫn input: ")
    input_folder = input().strip()
    print("Nhập đường dẫn output: ")
    output_folder = input().strip()
    
    # Xử lý
    process_folder(input_folder, output_folder)
    print("Hoàn thành!")
