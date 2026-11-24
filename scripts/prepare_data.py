from pathlib import Path
import hashlib, urllib.request, zipfile, re, sys

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
ARCHIVE = RAW / "26_29_09_2017_KCL.zip"
URL = "https://zenodo.org/records/2867216/files/26_29_09_2017_KCL.zip?download=1"
EXPECTED_MD5 = "98c51bdd2b092b93f8bb038dea4505fa"
PATTERN = re.compile(r"^(ID\d+)_(pd|hc)_(\d+)_(\d+)_(\d+)\.wav$", re.I)

def md5(path):
    h=hashlib.md5()
    with open(path,"rb") as f:
        for chunk in iter(lambda:f.read(1024*1024),b""): h.update(chunk)
    return h.hexdigest()

def main():
    RAW.mkdir(parents=True,exist_ok=True)
    wavs=list(RAW.rglob("*.wav"))
    if not wavs:
        print("Downloading MDVR-KCL from Zenodo (~606 MB). This is a one-time download.")
        try:
            req=urllib.request.Request(URL,headers={"User-Agent":"NeuroMargin-research/2.0"})
            with urllib.request.urlopen(req,timeout=60) as response, open(ARCHIVE,"wb") as out:
                total=int(response.headers.get("Content-Length",0))
                got=0
                while True:
                    chunk=response.read(1024*1024)
                    if not chunk: break
                    out.write(chunk); got+=len(chunk)
                    if total and got%(20*1024*1024)<1024*1024:
                        print(f"Downloaded {got/1024/1024:.0f} / {total/1024/1024:.0f} MB")
        except Exception as e:
            print("Dataset download failed:",e)
            print("Download manually from:",URL)
            return 2
        actual=md5(ARCHIVE)
        if actual != EXPECTED_MD5:
            print("Archive checksum mismatch. Expected",EXPECTED_MD5,"got",actual)
            return 2
        print("Archive checksum verified. Extracting...")
        with zipfile.ZipFile(ARCHIVE) as z: z.extractall(RAW)
    wavs=list(RAW.rglob("*.wav"))
    valid=[p for p in wavs if PATTERN.match(p.name)]
    print(f"Found {len(wavs)} WAV files; {len(valid)} filenames have recognized participant labels.")
    if not valid:
        print("No files matched expected ID##_pd/hc_...wav naming. Inspect extracted archive.")
        return 2
    labels={m.group(2).lower() for p in valid if (m:=PATTERN.match(p.name))}
    if labels != {"pd","hc"}:
        print("Both PD and HC classes are required; found",labels)
        return 2
    print("Dataset is ready:", RAW)
    return 0

if __name__=="__main__": sys.exit(main())
