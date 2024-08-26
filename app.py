"""
.pth, .pb파일을 torch, onnxruntime, tensorrt 환경에서 돌렸을 때 
차이점을 비교하기 위한 웹 사이트 
"""
import streamlit as st
import tempfile
import os


# Load model components 
from module.model_utils.convert_torch_to_onnx import export_to_onnx

# CSS components
from module.css_utils.button import button
st.markdown(button(), unsafe_allow_html=True)


# Sidebar 
with st.sidebar: 
    # 모델 파일 업로드 
    # 가능한 타입: .pth, .pb
    uploaded_file = st.file_uploader("", type=['pth', 'pb'])

    if uploaded_file is not None:
        file_name = str(uploaded_file.name) 

        st.success(f"{file_name} is uploaded", icon="✅")
        
        # 임시 파일 저장
        with tempfile.NamedTemporaryFile(delete=False) as temp_file:
            temp_file.write(uploaded_file.read())
            temp_file.flush()
            temp_file_path = temp_file.name

        st.write("")
        st.write("")
        st.write("")

        # .onnx로 모델 변환 
        if file_name.endswith('.pth'):
            file_name = file_name[:-4]
            onnx_file_path = export_to_onnx(temp_file_path, file_name)

            # .onnx 파일 다운로드
            with open(onnx_file_path, "rb") as onnx_file:
                st.download_button(
                    label=".onnx 파일 다운로드",
                    data=onnx_file,
                    file_name=os.path.basename(onnx_file_path),
                    mime="application/octet-stream"
                )
        else: 
            st.write("tensorflow 모델")
        

    # 파일이 업로드 되지 않았을 때, 경고 메시지 출력 
    else: 
        st.info(
            """
            모델 파일을 업로드 해주세요.  
            ( .pth / .pb )
            """, 
            icon="ℹ️"
        )