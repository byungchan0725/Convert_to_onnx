import streamlit as st
import tempfile
import os
from PIL import Image

# Load model components
from module.model_utils.convert.convert_torch_to_onnx import export_to_onnx
from module.model_utils.prediction.predict_single_image import show_image_with_acc

# CSS components
from module.css_utils.gap import gap_1
from module.css_utils.button import button
st.markdown(button(), unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    option = st.selectbox(
        "모드를 선택해주세요.",
        (".onnx 변환", "추론 속도 비교"),
    )
    
    if option == ".onnx 변환":
        upload_model_file = st.file_uploader("", type=['pth', 'pb'])

        if upload_model_file is not None:
            file_name = str(upload_model_file.name)
            st.success(f"{file_name} is uploaded", icon="✅")

            # 임시 파일 저장
            with tempfile.NamedTemporaryFile(delete=False) as temp_file:
                temp_file.write(upload_model_file.read())
                temp_file.flush()
                model_file = temp_file.name 

            gap_1() 

            if file_name.endswith('.pth'):
                """
                마지막 확장자가 .pth라면, torch 형식으로 로드하여 변환
                """
                file_name = file_name[:-4]  # 뒤에 확장자 자르기
                onnx_file_path = export_to_onnx(model_file, file_name)

                # 파일 다운로드 버튼 
                with open(onnx_file_path, "rb") as onnx_file:
                    st.download_button(
                        label=".onnx 파일 다운로드",
                        data=onnx_file,
                        file_name=os.path.basename(onnx_file_path),
                        mime="application/octet-stream"
                    )
            else:
                """ .pb 데이터는 미완성 """
                st.write("tensorflow 모델은 개발중 입니다.")
        else:
            st.info("(.pth .pb) 파일을 업로드 해주세요", icon="ℹ️")

    # -------------------- 추론 ---------------------
    else:
        # model_file: 모델 파일 업로드 
        # image_file: 이미지 파일 업로드 
        upload_model_file = st.file_uploader("모델 파일을 넣어주세요.", type=['onnx', 'pth', 'pb'])
        if upload_model_file is not None:
            file_name = str(upload_model_file.name)
            st.success(f"{file_name} is uploaded", icon="✅")

            # 임시 파일 저장
            with tempfile.NamedTemporaryFile(delete=False) as temp_file:
                temp_file.write(upload_model_file.read())
                temp_file.flush()
                model_file = temp_file.name

            gap_1() 

        upload_image_file = st.file_uploader("추론할 이미지를 넣어주세요.", type=['jpg', 'jpeg', 'png'])
        if upload_image_file is not None:
            file_name = upload_image_file.name
            st.success(f"{file_name} is uploaded", icon="✅")

            image_file = Image.open(upload_image_file)

            # 임시 파일 저장
            with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as temp_file:
                image_file.save(temp_file, format='PNG')
                image_file = temp_file.name
        else:
            image_file = None
            st.info("(.jpg .jpeg .png) 파일을 업로드 해주세요.", icon="ℹ️")

if 'image_file' in locals() and image_file and option == "추론 속도 비교":
    st.image(image_file, width=500)
    st.write("위 이미지에 대하여 추론을 진행합니다.")

    col1, col2, col3 = st.columns([10, 10, 10])
    test_ = show_image_with_acc(model_file, image_file)
    st.write(test_)
else:
    pass
