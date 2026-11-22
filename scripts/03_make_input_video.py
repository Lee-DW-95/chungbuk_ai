"""
Wikimedia Commons의 CC 라이선스 영상 3개에서 구간을 잘라 5초짜리 테스트 영상을 만든다.
(비행기 1.2초 + 독수리 1.9초 + 까마귀 1.9초, 1920x1080, 30fps)

원본 영상 (data/videos_raw/ 에 저장):
- airplane_hkia.webm : "Airplane Landing at South Runway, HKIA 20210215", IamCristYe, CC BY-SA 4.0
- eagle_sea.webm     : "Zittende en vliegende zeearenden van verschillende leeftijden", Marc Plomp, CC BY-SA 3.0 NL
- crow_park.webm     : "Jungle crow (Corvus macrorhynchos) in a park", The Nature Box, CC BY 4.0
"""
import subprocess
from pathlib import Path

import imageio_ffmpeg

RAW = Path("data/videos_raw")
OUT = Path("videos/input_5s.mp4")
CLIPS = [  # (파일, 시작초, 길이초)
    # 비행기가 크게, 가림 없이 하늘 배경으로 지나가는 구간 (이후에는 어두운 산 앞으로 들어가 대비가 약해짐)
    ("airplane_hkia.webm", 22.6, 1.2),
    ("eagle_sea.webm", 212.5, 1.9),
    ("crow_park.webm", 10.0, 1.9),
]

ff = imageio_ffmpeg.get_ffmpeg_exe()
OUT.parent.mkdir(parents=True, exist_ok=True)
cmd = [ff, "-y", "-loglevel", "error"]
for name, ss, t in CLIPS:
    cmd += ["-ss", str(ss), "-t", str(t), "-i", str(RAW / name)]
# 비율이 16:9가 아닌 영상(독수리 4:3)은 검은 띠 대신 같은 영상을 흐리게 확대한 배경으로 채운다
norm = "".join(
    f"[{i}:v]setsar=1,split[a{i}][b{i}];"
    f"[a{i}]scale=1920:1080:force_original_aspect_ratio=increase,crop=1920:1080,boxblur=40:2,eq=brightness=-0.08[bg{i}];"
    f"[b{i}]scale=1920:1080:force_original_aspect_ratio=decrease[fg{i}];"
    f"[bg{i}][fg{i}]overlay=(W-w)/2:(H-h)/2,fps=30,setsar=1[v{i}];"
    for i in range(len(CLIPS))
)
concat = "".join(f"[v{i}]" for i in range(len(CLIPS))) + f"concat=n={len(CLIPS)}:v=1:a=0[out]"
cmd += ["-filter_complex", norm + concat, "-map", "[out]", "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "18", str(OUT)]
subprocess.run(cmd, check=True)
print(f"[done] {OUT}")
