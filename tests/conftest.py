from typing import Dict, List, Tuple

import pytest


@pytest.fixture
def vn_dict() -> List[str]:
    return ['anh', 'rất', 'ngại']


@pytest.fixture
def abbr_dict() -> Dict[str, List[str]]:
    return {'ATTT': ['An toàn thông tin'], 'ĐT': ['Đội tuyển', 'Đào tạo', 'Điện thoại']}



@pytest.fixture
def normalize_examples() -> List[str]:
    return [
        'Vụ cô gái bị sát hại ở chợ đầu mối Thủ Đức: Khởi tố chồng nữ hung thủ',
        'Theo hãng tin Reuters, cửa khẩu Rafah là điểm vượt biên duy nhất có thể vào Sinai của 2,3 triệu người ở dải Gaza.',
        'Chiều dài cầu vượt tăng 260m phù hợp với việc phối hợp ga Dân Chủ của dự án tuyến metro số 2. Công trình dự kiến khởi công cuối năm 2024 hoặc đầu 2025 và hoàn thành trong năm 2025.',
        'Từ 1-7-2024: Bãi bỏ mức lương cơ sở và hệ số lương',
        'Bà Trịnh Thị Thuỷ thông tin: "Trung tâm HLTTQG Hà Nội không đủ điều kiện diện tích đảm bảo cho hơn 40 đội tuyển trẻ tập luyện hàng năm (khoảng 1 nghìn người).',
    ]


@pytest.fixture
def nsw_detect_examples() -> List[List[str]]:
    return [
        ['ĐT', 'Việt', 'Nam', 'giành', 'huy', 'chương', 'ở', 'nhiều', 'bộ', 'môn'],
        ['Hè', 'đến', ',', 'bao', 'kỉ', 'niệm', 'lại', 'ùa', 'về', '...'],
        ['This', 'is', 'how', 'it', 'will', 'end'],
        ['Theo', 'diễn', 'biến', 'vụ', 'việc', ',', 'ngày', '16.3.2023', ',', 'cơ', 'quan', 'chức', 'năng', 'phát', 'hiện', '4', 'nữ', 'tiếp', 'viên', 'hàng', 'không', 'vận', 'chuyển', 'ma', 'túy', 'từ', 'Pháp', 'về', 'Việt', 'Nam', '.']
    ]

@pytest.fixture
def tokenize_examples() -> List[Tuple[str, List[str]]]:
    return [
        (
            'từ11-18 là abc@gmail.com những .18 người  89.hôm nay adu.com.vn đã gọi được,19-20 là người hôm nay http://abc.com/anh-rất-ngại chưa gọi được ⇒ ngày hôm sau vẫn giữ ưu tiên người từ 1-10 ,sau đó là 19-20,sau đó là 11-18?',
            ['từ', '11-18', 'là', 'abc@gmail.com', 'những', '.', '18', 'người', '89', '.', 'hôm', 'nay', 'adu.com.vn', 'đã', 'gọi', 'được', ',', '19-20', 'là', 'người', 'hôm', 'nay', 'http://abc.com/anh-rất-ngại', 'chưa', 'gọi', 'được', '⇒', 'ngày', 'hôm', 'sau', 'vẫn', 'giữ', 'ưu', 'tiên', 'người', 'từ', '1-10', ',', 'sau', 'đó', 'là', '19-20', ',', 'sau', 'đó', 'là', '11-18', '?']
        ),
        (
            'Như tờ The Sun, cây viết Kealan Hughes để luôn chữ “Clown” – “Gã hề” ngay đầu tiêu đề và sử dụng một từ rất “mạnh” khác là “Embarrassing” – “Đáng xấu hổ” đi kèm với tên của Bruno Fernandes.',
            ['Như', 'tờ', 'The', 'Sun', ',', 'cây', 'viết', 'Kealan', 'Hughes', 'để', 'luôn', 'chữ', 'Clown', '–', 'Gã', 'hề', 'ngay', 'đầu', 'tiêu', 'đề', 'và', 'sử', 'dụng', 'một', 'từ', 'rất', 'mạnh', 'khác', 'là', 'Embarrassing', '–', 'Đáng', 'xấu', 'hổ', 'đi', 'kèm', 'với', 'tên', 'của', 'Bruno', 'Fernandes', '.']
        ),
        (
            '3 người đàn ông "sức dài vai rộng" nhưng mang tiếng "ăn bám nhà vợ": Người cuối cùng là thất vọng nhất',
            ['3', 'người', 'đàn', 'ông', 'sức', 'dài', 'vai', 'rộng', 'nhưng', 'mang', 'tiếng', 'ăn', 'bám', 'nhà', 'vợ', ':', 'Người', 'cuối', 'cùng', 'là', 'thất', 'vọng', 'nhất']
        ),
        (
            '"Hỏa tiễn" chống hạm P-800 Oniks: Tốc độ siêu âm, diệt mục tiêu cách 600 km',
            ['Hỏa', 'tiễn', 'chống', 'hạm', 'P-800', 'Oniks', ':', 'Tốc', 'độ', 'siêu', 'âm', ',', 'diệt', 'mục', 'tiêu', 'cách', '600', 'km']
        ),
        (
            'Bảng giá MacBook tháng 3/2023: Giảm toàn bộ, lên tới 26%.',
            ['Bảng', 'giá', 'MacBook', 'tháng', '3/2023', ':', 'Giảm', 'toàn', 'bộ', ',', 'lên', 'tới', '26%', '.']
        ),
        (
            'Ông PhạmMinh cho biết rất yên tâm khi ban tổ chức giải đã lo chu đáo cho đội chế độ ăn uống, nghỉ ngơi tại khách sạn tại Q.5. "TP.HCM thì quá hiện đại, tiện nghi.',
            ['Ông', 'Phạm', 'Minh', 'cho', 'biết', 'rất', 'yên', 'tâm', 'khi', 'ban', 'tổ', 'chức', 'giải', 'đã', 'lo', 'chu', 'đáo', 'cho', 'đội', 'chế', 'độ', 'ăn', 'uống', ',', 'nghỉ', 'ngơi', 'tại', 'khách', 'sạn', 'tại', 'Q.5', '.', 'TP.HCM', 'thì', 'quá', 'hiện', 'đại', ',', 'tiện', 'nghi', '.']
        ),
        (
            'có vị trí địa lý hết sức thuận lợi khi nằm trên trục đường nối cao tốc Hà Nội – Hải Phòng và Cầu Giẽ -Ninh Bình, thuận tiện cho việc di chuyển đến các sân bay, cảng biển lớn.',
            ['có', 'vị', 'trí', 'địa', 'lý', 'hết', 'sức', 'thuận', 'lợi', 'khi', 'nằm', 'trên', 'trục', 'đường', 'nối', 'cao', 'tốc', 'Hà', 'Nội', '–', 'Hải', 'Phòng', 'và', 'Cầu', 'Giẽ', '-', 'Ninh', 'Bình', ',', 'thuận', 'tiện', 'cho', 'việc', 'di', 'chuyển', 'đến', 'các', 'sân', 'bay', ',', 'cảng', 'biển', 'lớn', '.']
        ),
        (
            '"Hóa đơn tiền điện của chúng tôi tăng gần gấp đôi có phải do phải trả 627 kWh điện này theo giá của bậc 6 không?", ông N.T.T thắc mắc.',
            ['Hóa', 'đơn', 'tiền', 'điện', 'của', 'chúng', 'tôi', 'tăng', 'gần', 'gấp', 'đôi', 'có', 'phải', 'do', 'phải', 'trả', '627', 'kWh', 'điện', 'này', 'theo', 'giá', 'của', 'bậc', '6', 'không', '?', ',', 'ông', 'N.T.T', 'thắc', 'mắc', '.']
        ),
        (
            '-26',
            ['-26']
        ),
        (
            'Như nhìn nhận của một số nhà chuyên môn, hiện nay nếu muốn chinh phục người nghe, bên cạnh kỹ thuật tốt, giọng hát nội lực, còn phải chú trọng đến những yếu tố tiếp cận đại chúng (phải có ca khúc riêng, âm nhạc phù hợp xu hướng, hình ảnh/vũ đạo thu hút…). "Có thể nói, với Về với em, Võ Hạ Trâm thành công khi đã bước ra vùng an toàn của mình.',
            ['Như', 'nhìn', 'nhận', 'của', 'một', 'số', 'nhà', 'chuyên', 'môn', ',', 'hiện', 'nay', 'nếu', 'muốn', 'chinh', 'phục', 'người', 'nghe', ',', 'bên', 'cạnh', 'kỹ', 'thuật', 'tốt', ',', 'giọng', 'hát', 'nội', 'lực', ',', 'còn', 'phải', 'chú', 'trọng', 'đến', 'những', 'yếu', 'tố', 'tiếp', 'cận', 'đại', 'chúng', 'phải', 'có', 'ca', 'khúc', 'riêng', ',', 'âm', 'nhạc', 'phù', 'hợp', 'xu', 'hướng', ',', 'hình', 'ảnh/vũ', 'đạo', 'thu', 'hút', '…', '.', 'Có', 'thể', 'nói', ',', 'với', 'Về', 'với', 'em', ',', 'Võ', 'Hạ', 'Trâm', 'thành', 'công', 'khi', 'đã', 'bước', 'ra', 'vùng', 'an', 'toàn', 'của', 'mình', '.']        
        )
    ]
