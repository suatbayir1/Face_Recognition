import cv2
import imageio

print(cv2.__version__)

# cascade'ler filtre serileridir. Bunlar objeyi tespit etmek için arka arakaya uygulanır.
face_cascade = cv2.CascadeClassifier('haarcascade-frontalface-default.xml')
eye_cascade = cv2.CascadeClassifier('haarcascade-eye.xml')

# tarama yapılacak resmi parametre olarak veriyoruz
def detect(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # renkli resmi gri resme çevir
    #resimde tespit edilen yüzlerin koordinatını almamız gerekiyor
    faces = face_cascade.detectMultiScale(gray, 1.3, 5) # x,y,w,h dikdörtgenlerini tutan tuple'lar var (tespit edilen yüzler)

    for (x, y, w, h) in faces:   #bu döngü sayesinde her yüze tek tek dikdörtgen çiziyoruz. (x,y,w,h bulunan i. yüzün bilgileri)
        cv2.rectangle(frame, (x,y), (x+w, y+h), (255,0,0), 2)  # 1.parametre resim, 2.parametre sol üst köşenin koordinatları 3.parametre sağ alt köşenin koordinatları 4.parametre dikdörtgenin renklerini 5.parametre dikdörtgenin kalınlıgını belirtir
        # gri resimde göz bulundugunda bunu orjinal resimde de çizecek
        gray_face = gray[y:y+h, x:x+w] # gri resim üzerinde yüzün bulundugu dikdörtgeni alıyoruz
        color_face = frame[y:y+h, x:x+w] # orjinal(renkli) resim üzerinde yüzün bulundugu alan
        eyes = eye_cascade.detectMultiScale(gray_face, 1.1, 3) # sadece yüzün oldugu yerde göz arıyoruz
        for (ex, ey, ew, eh) in eyes:
            cv2.rectangle(color_face, (ex, ey), (ex+ew, ey+eh), (0,255,0), 2) #bulunan gözü çiz
    return frame  #frame içinde dikdörtgenler halinde yüz ve göz gider


reader = imageio.get_reader('video.mp4') #videoyu oku
fps = reader.get_meta_data()['fps'] # fps oranını al (giriş ve çıkış videoları aynı hızda olsun diye)
writer = imageio.get_writer('output.mp4', fps=fps) #tespit edilen nesneyi videoya aktar

# reader ile aldıgımız video içerisindeki her frame'de dön. i'yi sayaç olarak kullan
for i, frame in enumerate(reader):
    frame = detect(frame) # videodan alınan her frame için yukarıdaki fonksiyonda yüz ve göz tanıma yap
    writer.append_data(frame) #bulunan yüz ve gözü çıkış videosuna yükle
    print(i) # kaçıncı framedeyiz takip ediyoruz

writer.close() #videoyu sonlandır.