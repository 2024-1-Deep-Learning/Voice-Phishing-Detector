import os
import json
import pandas as pd
import re

# 한글 문자만 추출하는 함수
def extract_korean(text):
    """주어진 텍스트에서 한글 문자와 공백만 남기고 제거합니다."""
    return re.sub(r'[^가-힣\s]', '', text)

if __name__ == "__main__":
    # 기본 경로 설정
    base_path = '전화망데이터/'

    # 결합된 데이터를 저장할 리스트
    combined_data = []

    # 폴더 번호 범위 설정
    start_index = 1
    end_index = 38329

    # 모든 폴더를 순회하며 JSON 파일 처리
    for i in range(start_index, end_index + 1):
        folder_name = f'S{i:06d}'
        json_file_path = os.path.join(base_path, folder_name, f'{folder_name}.json')
        
        # 폴더와 JSON 파일이 존재하는 경우에만 처리
        if os.path.exists(json_file_path):
            try:
                with open(json_file_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    dialogs = data['dataSet']['dialogs']
                    combined_text = ' '.join([dialog['text'] for dialog in dialogs])
                    korean_text = extract_korean(combined_text)
                    combined_data.append({
                        'file': folder_name,
                        'transcript': korean_text
                    })
            except Exception as e:
                print(f"파일 {json_file_path} 처리 중 오류 발생: {e}")

    # 데이터프레임 생성
    df = pd.DataFrame(combined_data)

    # 엑셀 파일로 저장
    output_path = 'data/combined_transcriptions.xlsx'
    df.to_excel(output_path, index=False)

    print(f"결합된 텍스트가 {output_path}에 저장되었습니다.")
