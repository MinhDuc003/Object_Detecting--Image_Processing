import streamlit as st
from PIL import Image

st.set_page_config(
    page_title="Trang Chủ",
    page_icon="👋",
)

logo = Image.open('images\logo.png')
st.image(logo, width=800)

st.title("Website Xử lý ảnh")
st.caption("Thực hiện bởi: Huỳnh Minh Đức và Phạm Ngọc Đăng Khoa")
st.caption("Giảng viên hướng dẫn: ThS. Trần Tiến Đức")
st.caption("Lớp Xử lý ảnh: DIPR430685_23_2_04CLC")

left, right = st.columns(2)



st.markdown(
    """
    ### Thông tin liên hệ
    - Email: 21110168@student.hcmute.edu.vn hoặc 21110214@student.hcmute.edu.vn
    - Lấy source code tại [đây](https://discuss.streamlit.io)
    """
)
