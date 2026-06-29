# Firefly Projesi: PID Guidance Kontrolcüleri ve Simulation Bridge Kullanım Rehberi

Bu belge, Firefly projesinin **yönlendirme (guidance)** ve **simülasyon (simulation)** mimarilerini, `src/python/guidance/` altındaki PID dosyalarının detaylı işlevlerini ve `src/python/simulation/sim_bridge.py` dosyasının simülasyon sistemindeki rolünü detaylıca açıklamak amacıyla hazırlanmıştır.

---

## 1. Genel Mimari ve Konfigürasyon

Firefly otonom uçuş sistemi, uçuş dinamiklerini kontrol etmek için **PID (Proportional-Integral-Derivative)** kontrolcülerini ve durum yönetimi için kaskat (köprülenmiş) kontrol yapılarını kullanır. Tüm PID parametreleri, sınır değerler ve uçuş modları tek bir merkezden (`config/pid_params.yaml`) yapılandırılır.

---

## 2. Guidance Modülleri (`src/python/guidance/`)

`guidance` klasörü, İHA'nın hedefe kilitlenmesini, irtifasını, hızını ve yönelmesini sağlayan temel kontrol algoritmalarını içerir.

---

### 2.1 `pid_core.py` — Temel Matematik ve PID Sınıfı

`pid_core.py`, tüm kontrolcülerin temelini oluşturan matematiksel yardımcı araçları ve genel PID yapısını tanımlar.

#### **İçerik ve Sınıflar:**

1. **`PID` Sınıfı:**
   - **Görevi:** İstenen hedef değer ile mevcut durum arasındaki hatayı (error) kullanarak Orantısal ($K_p$), İntegral ($K_i$) ve Türev ($K_d$) bileşenleri üzerinden düzeltme sinyali üretir.
   - **Özellikleri:**
     - **Anti-Windup (Integral Limiting):** `integral_limit` parametresi ile integral birikmesinin aşırı artıp kontrol kaybına yol açması engellenir.
     - **Komut Kırpma (Clamping):** Çıktı değeri `clamp=(min, max)` aralığına sınırlandırılır.
     - **İlk Çalışma Yumuşatması (`first_run`):** Türev bileşeninin ilk adımda ani sıçrama yapmasını önler.

2. **`SlewLimiter` Sınıfı:**
   - **Görevi:** Ani açı veya hız komut değişimlerini zaman içinde yumuşatarak (slew rate) Donanım/İHA mekaniğine ani yük binmesini engeller.
   - **Çalışma Mantığı:** Saniyede izin verilen maksimum değişim miktarını (`max_change_per_sec`) kullanarak hedef değere kademeli olarak yaklaşır.

3. **Yardımcı Fonksiyonlar:**
   - **`wrap_180(angle)`:** Verilen açı değerini $-180^\circ$ ile $+180^\circ$ aralığına getirir. Yönelme (yaw) hesaplamalarında en kısa dönüş açısını bulmak için kullanılır.
   - **`clamp(val, lo, hi)`:** Herhangi bir sayısal değeri belirlenen alt (`lo`) ve üst (`hi`) sınırlar arasında tutar.

---

### 2.2 `pid_pitch.py` — Pitch ve Hız Kontrolcüsü (`PitchController`)

İHA'nın boylamasına eğim (pitch) açısını ve buna bağlı olarak ileri yönlü hızını/mesafesini kontrol eden modüldür.

#### **Çalışma Mantığı ve Önemli Özellikler:**

- **Çift Halkalı / Kaskat Yapı:** Dış halkada hız PID'si (`speed_pid`) ve iç halkada pitch açısı PID'si (`inner_pid`) bulunur.
- **Kamera Açısı Kompanzasyonu (`camera_tilt`):** İHA'nın hedefi kamerayla takip edebilmesi için kamera tilt açısı (örneğin $35^\circ$) elevasyon hesabına dahil edilir (`camera_center_pitch = target_elevation - camera_tilt`).
- **Yaw Hizalanması Ölçeklemesi (`yaw_scale`):** İHA henüz hedefe doğru dönmemişken ileri doğru hızlanıp rotadan sapmasını engellemek için pitch komutu yaw hatasının kosinüsü ile ölçeklenir (`math.cos(math.radians(yaw_error))`). Yön hedefe tam hizalandığında ($\cos(0^\circ) = 1$) tam pitch uygulanır; dik olduğunda ($\cos(90^\circ) = 0$) pitch sıfırlanır.
- **Normalize Çıktı:** Hesaplanan derece cinsinden açıyı otopilotun / SDK'nın beklediği `[-1.0, 1.0]` aralığına ölçekler.

#### **Giriş ve Çıkış Parametreleri (`calculate` fonksiyonu):**
- **Girdiler:** `state` (uçuş modu), `target_speed_cmd` (hedef hız), `drone_speed` (mevcut hız), `target_elevation` (hedef yükselim açısı), `drone_pitch` (mevcut pitch), `yaw_error` (yönelme hatası), `dt` (geçen süre).
- **Çıktılar:** `smooth_desired` (yumuşatılmış hedef pitch açısı), `pitch_cmd` (normalize edilmiş `[-1, 1]` pitch komutu).

---

### 2.3 `pid_yaw.py` — Yaw (Yönelme) Kontrolcüsü (`YawController`)

İHA'nın yatay düzlemde (heading/bearing) hedef nesneye veya koordinata kilitlenmesini sağlar.

#### **Çalışma Mantığı:**

1. İHA ile hedef arasındaki $X$ ve $Y$ mesafe farklarını (`dx`, `dy`) kullanarak `math.atan2(dy, dx)` ile hedef kerteriz açısını (`target_bearing`) derece cinsinden hesaplar.
2. `wrap_180` fonksiyonu ile İHA'nın mevcut yaw açısı arasındaki en kısa dönme açısını (`yaw_error`) tespit eder.
3. İç halka yaw PID'si aracılığıyla bu hatayı sıfırlayacak normalize yaw sapma komutunu (`yaw_cmd`) üretir ve `[-1.0, 1.0]` aralığına sınırlar.

#### **Giriş ve Çıkış Parametreleri (`calculate` fonksiyonu):**
- **Girdiler:** `dx`, `dy` (hedef-İHA koordinat farkları), `drone_yaw` (İHA'nın mevcut yönü), `dt` (zaman adımı).
- **Çıktılar:** `yaw_error` (açısal hata °), `yaw_cmd` (normalize edilmiş `[-1, 1]` yaw komutu).

---

### 2.4 `pid_throttle.py` — Throttle ve İrtifa Kontrolcüsü (`ThrottleController`)

İHA'nın dikey eksendeki ($Z$) konumunu korumasını ve hedef ile arasındaki yükseklik farkını sıfırlamasını sağlayan dikey kontrol modülüdür.

#### **Çalışma Mantığı:**

- **Base Throttle (Askıda Kalma / Hover Offseti):** Otopilot SDK'sında $0.0$ değeri askıda kalmayı (hover) temsil eder. `base_throttle` bu temel değer üzerine inşa edilir.
- **İrtifa Düzeltmesi:** Hedef ile İHA arasındaki yükseklik farkı metreye dönüştürülür ve `alt_pid` (irtifa PID'si) üzerinden bir düzeltme kuvveti hesaplanır.
- **Limit Koruması:** Hesaplanan gaz değeri `min_throttle` (örneğin -0.8) ve `max_throttle` (örneğin 0.8) sınırları arasında kısıtlanır.

#### **Giriş ve Çıkış Parametreleri (`calculate` fonksiyonu):**
- **Girdiler:** `dz` (hedef $Z$ - İHA $Z$ yükseklik farkı), `dt` (zaman adımı).
- **Çıktı:** `target_throttle` (normalize `[-1, 1]` veya konfigürasyona uygun gaz komutu).

---

## 3. Simulation Modülü (`src/python/simulation/sim_bridge.py`)

`sim_bridge.py`, yukarıda açıklanan tüm guidance kontrolcülerini, mod yöneticisini (`ModeManager`) ve İHA SDK'sını bir araya getirerek gerçek zamanlı simülasyon döngüsünü çalıştıran **ana yürütücü (entrypoint)** dosyadır.

---

### 3.1 `sim_bridge.py` Yapısı ve Çalışma Akışı

Dosya, Python arka planını Gazebo / ROS2 / PX4 simülasyon ortamına bağlar ve döngüsel olarak telemetry okuyup komut üretir.

#### **Adım Adım Simülasyon Döngüsü:**

1. **Konfigürasyon ve Başlatma (`run_simulation`):**
   - `config/pid_params.yaml` dosyasından konfigürasyonu yükler.
   - `ModeManager`, `PitchController`, `ThrottleController` ve `YawController` nesnelerini başlatır.
   - `drone.connect()` ile simülatör bağlantısını kurar ve `drone.set_arm(True)` ile motorlara güç verir.

2. **Gerçek Zamanlı Zaman Adımı (`dt`) Hesaplama:**
   - Sabit bir zaman adımı yerine sistem saatini (`time.time()`) ölçerek gerçek geçen süreyi hesaplar ve `dt` değerini $0.01$ s ile $0.1$ s arasında sınırlar.

3. **Telemetri Verilerinin Okunması:**
   - `drone.get_telemetry()` üzerinden İHA ve Hedefin konum (`position`), yönelme (`rotation`) ve hız (`speed`) verilerini çeker. Veri henüz gelmemişse döngüyü atlar.

4. **Vektörel ve Geometrik Hesaplamalar:**
   - Hedef ile İHA arasındaki mesafe farkları ($dx, dy, dz$) bulunur.
   - 2D yatay mesafe ($\sqrt{dx^2+dy^2}$) ve 3D toplam mesafe ($\sqrt{dx^2+dy^2+dz^2}$) hesaplanır.
   - Hedefe bakış yükselim açısı (`target_elevation`) $\arctan(dz / dist\_2d)$ ile dereceye çevrilir.

5. **Kontrol Kararlarının Üretilmesi (Adım Sırası):**
   - **Adım 1 (Yaw):** `yaw_ctrl.calculate(...)` çağrılarak yaw hatası ve normalize dönme komutu alınır.
   - **Adım 2 (Mod Güncelleme):** `mode_mgr.update(...)` çağrılarak uçuş modu (`KALKIS`, `SEYIR`, `TAKIP`) ve hedef hız belirlenir.
   - **Adım 3 (Pitch):** `pitch_ctrl.calculate(...)` ile hız hatası ve elevasyon dikkate alınarak pitch açısı hesaplanır ve yaw hizasına göre ölçeklenmiş komut üretilir.
   - **Adım 4 (Throttle):** `throttle_ctrl.calculate(...)` ile irtifa koruma/yaklaşma gaz komutu üretilir.
   - **Adım 5 (Roll):** Angle Mode / Otopilot seviyesinde yatay denge korunduğu için roll komutu $0.0$ olarak sabit tutulur.

6. **Donanıma / Simülatöre Komut Gönderimi:**
   - `drone.set_control_surfaces(throttle, pitch, roll, yaw, auto_wing)` fonksiyonu ile hesaplanan tüm komutlar güvenli sınırlar (`clamp`) içinde tutularak simülatöre iletilir.

7. **Konsol Bilgilendirmesi ve Güvenlik:**
   - Her döngüde anlık mod, mesafe, irtifa farkı, hız ve kontrol komutları konsola yazdırılır.
   - Kullanıcı simülasyonu sonlandırdığında (`Ctrl+C`) veya hata oluştuğunda `finally` bloğu çalışarak motorları disarm eder (`drone.set_arm(False)`) ve bağlantıyı güvenli şekilde kapatır.

---

## 4. Özet Kullanım Tablosu

| Modül / Dosya | Ana Sınıf / Fonksiyon | Temel Görevi | Girdi | Çıktı |
| :--- | :--- | :--- | :--- | :--- |
| **`pid_core.py`** | `PID`, `SlewLimiter` | Algoritma ve matematik altyapısı | Hata (Error), dt | Düzeltme Miktarı |
| **`pid_pitch.py`** | `PitchController` | Hız & Pitch açısı kontrolü | Mod, Hızlar, Elevasyon, Yaw Error | `smooth_desired`, `pitch_cmd` |
| **`pid_yaw.py`** | `YawController` | Hedefe yönelme (Bearing) | $dx, dy$, Mevcut Yaw | `yaw_error`, `yaw_cmd` |
| **`pid_throttle.py`** | `ThrottleController` | İrtifa ve Gaz kontrolü | $dz$ (İrtifa farkı), dt | `target_throttle` |
| **`sim_bridge.py`** | `run_simulation()` | Simülasyon döngüsü yürütücü | Telemetri & Konfigürasyon | Donanım Komutları (`set_control_surfaces`) |
