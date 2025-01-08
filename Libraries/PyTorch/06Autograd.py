import torch

# torch.autograd ile Otomatik Farklılaştırma(Automatic Differentiation(AutoGrad))
# Sinir ağlarını eğitirken en sık kullanılan algoritma geri yayılımdır.
# Bu algoritmada, parametreler (model ağırlıkları) kayıp fonksiyonunun verilen parametreye göre gradyanına göre ayarlanır.
# Bu gradyanları hesaplamak için PyTorch, torch.autograd adlı yerleşik bir farklılaştırma motoruna sahiptir.
# Herhangi bir hesaplama grafiği için gradyanın otomatik olarak hesaplanmasını destekler.
# Girdi x, w ve b parametreleri ve bazı kayıp fonksiyonları ile en basit tek katmanlı sinir ağını düşünün. PyTorch'ta aşağıdaki şekilde tanımlanabilir:

x = torch.ones(5)  # input tensor
y = torch.zeros(3)  # expected output
w = torch.randn(5, 3, requires_grad=True)
b = torch.randn(3, requires_grad=True)
z = torch.matmul(x, w) + b
loss = torch.nn.functional.binary_cross_entropy_with_logits(z, y)

# Tensörler, Fonksiyonlar ve Hesaplama Grafiği
# Bu kod Downloaded klasöründeki autograd resmindeki hesaplama grafiğini tanımlar:
# Bu ağda, w ve b optimize etmemiz gereken parametrelerdir.
# Bu nedenle, bu değişkenlere göre kayıp fonksiyonunun gradyanlarını hesaplayabilmemiz gerekir.
# Bunu yapmak için, bu tensörlerin requires_grad özelliğini ayarlıyoruz.
# Not: requires_grad değerini bir tensör oluştururken veya daha sonra x.requires_grad_(True) yöntemini kullanarak ayarlayabilirsiniz.

# Hesaplama grafiği oluşturmak için tensörlere uyguladığımız bir fonksiyon aslında Function sınıfının bir nesnesidir.
# Bu nesne, fonksiyonun ileri yönde nasıl hesaplanacağını ve ayrıca geriye doğru yayılma adımı sırasında türevinin nasıl hesaplanacağını bilir.
# Geriye doğru yayılma fonksiyonuna bir referans, bir tensörün grad_fn özelliğinde saklanır. Function hakkında daha fazla bilgiyi dokümantasyonda bulabilirsiniz.

print(f"Gradient function for z = {z.grad_fn}")
print(f"Gradient function for loss = {loss.grad_fn}")

# Gradyanların Hesaplanması
# Sinir ağındaki parametrelerin ağırlıklarını optimize etmek için, kayıp fonksiyonumuzun parametrelere göre türevlerini hesaplamamız gerekir.
# Bu türevleri hesaplamak için loss.backward()'ı çağırırız ve ardından değerleri w.grad ve b.grad'dan alırız:

loss.backward()
print(w.grad)
print(b.grad)

# Not 1: Yalnızca requires_grad özelliği True olarak ayarlanmış olan hesaplama grafiğinin yaprak düğümleri için grad özelliklerini elde edebiliriz.
# Grafiğimizdeki diğer tüm düğümler için gradyanlar mevcut olmayacaktır.
# Not 2: Performans nedenleriyle, belirli bir grafik üzerinde geriye doğru kullanarak gradyan hesaplamalarını yalnızca bir kez gerçekleştirebiliriz.
# Aynı grafik üzerinde birden fazla geriye doğru çağrı yapmamız gerekiyorsa, geriye doğru çağrıya retain_graph=True değerini geçirmemiz gerekir.

# Degrade İzlemeyi Devre Dışı Bırakma
# Varsayılan olarak, requires_grad=True olan tüm tensörler hesaplama geçmişlerini izler ve gradyan hesaplamasını destekler.
# Bununla birlikte, bunu yapmamız gerekmeyen bazı durumlar vardır.
# örneğin, modeli eğittiğimizde ve sadece bazı girdi verilerine uygulamak istediğimizde, yani sadece ağ üzerinden ileri hesaplamalar yapmak istediğimizde.
# Hesaplama kodumuzu torch.no_grad() bloğu ile çevreleyerek izleme hesaplamalarını durdurabiliriz:

z = torch.matmul(x, w) + b
print(z.requires_grad)

with torch.no_grad():
    z = torch.matmul(x, w) + b
print(z.requires_grad)

# Aynı sonucu elde etmenin bir başka yolu da tensör üzerinde detach() yöntemini kullanmaktır:
z = torch.matmul(x, w) + b
z_det = z.detach()
print(z_det.requires_grad)

# Degrade izlemeyi devre dışı bırakmak isteyebileceğiniz nedenler vardır:
# Sinir ağınızdaki bazı parametreleri dondurulmuş parametreler olarak işaretlemek için.
# Sadece ileri geçiş yaptığınızda hesaplamaları hızlandırmak için, çünkü gradyanları takip etmeyen tensörler üzerindeki hesaplamalar daha verimli olacaktır.

# Hesaplamalı Graflar hakkında daha fazla bilgi
# Kavramsal olarak, autograd verilerin (tensörler) ve yürütülen tüm işlemlerin (ortaya çıkan yeni tensörlerle birlikte) kaydını Fonksiyon nesnelerinden oluşan bir yönlendirilmiş asiklik grafikte (DAG) tutar.
# Bu DAG'da yapraklar girdi tensörleri, kökler ise çıktı tensörleridir.
# Bu grafiği köklerden yapraklara doğru izleyerek, zincir kuralını kullanarak gradyanları otomatik olarak hesaplayabilirsiniz.

# Bir ileri geçişte, autograd aynı anda iki şey yapar:
# Ortaya çıkan tensörü hesaplamak için istenen işlemi çalıştırır.
# DAG'de işlemin gradyan fonksiyonunu korur.

# DAG kökünde .backward() çağrıldığında geriye doğru geçiş başlar. autograd o zaman:
# her .grad_fn'den gradyanları hesaplar.
# bunları ilgili tensörün .grad niteliğinde biriktirir.
# zincir kuralını kullanarak yaprak tensörlere kadar yayılır.

# Not: PyTorch'ta DAG'lar dinamiktir Dikkat edilmesi gereken önemli bir husus, grafiğin sıfırdan yeniden oluşturulmasıdır;
# her .backward() çağrısından sonra autograd yeni bir grafiği doldurmaya başlar.
# Bu tam olarak modelinizde kontrol akışı ifadelerini kullanmanıza izin veren şeydir; gerekirse her yinelemede şekli, boyutu ve işlemleri değiştirebilirsiniz.

# İsteğe Bağlı Okuma: Tensör Gradyanları ve Jacobian Çarpımları
# Çoğu durumda, skaler bir kayıp fonksiyonumuz vardır ve bazı parametrelere göre gradyanı hesaplamamız gerekir.
# Ancak, çıktı fonksiyonunun keyfi bir tensör olduğu durumlar da vardır.
# Bu durumda, PyTorch gerçek gradyanı değil, Jacobian çarpımı olarak adlandırılan çarpımı hesaplamanıza izin verir.

inp = torch.eye(4, 5, requires_grad=True)
out = (inp + 1).pow(2).t()
out.backward(torch.ones_like(out), retain_graph=True)
print(f"First call\n{inp.grad}")
out.backward(torch.ones_like(out), retain_graph=True)
print(f"\nSecond call\n{inp.grad}")
inp.grad.zero_()
out.backward(torch.ones_like(out), retain_graph=True)
print(f"\nCall after zeroing gradients\n{inp.grad}")

# Aynı argümanla ikinci kez geriye doğru çağırdığımızda, gradyan değerinin farklı olduğuna dikkat edin.
# Bunun nedeni, geriye doğru yayılma yaparken PyTorch'un gradyanları biriktirmesidir, yani hesaplanan gradyanların değeri, hesaplama grafiğinin tüm yaprak düğümlerinin grad özelliğine eklenir.
# Doğru gradyanları hesaplamak istiyorsanız, daha önce grad özelliğini sıfırlamanız gerekir. Gerçek hayat eğitiminde bir optimizer bunu yapmamıza yardımcı olur.

# Not: Daha önce backward() fonksiyonunu parametresiz çağırıyorduk.
# Bu aslında backward(torch.tensor(1.0)) çağrısına eşdeğerdir ve sinir ağı eğitimi sırasında kayıp gibi skaler değerli bir fonksiyon olması durumunda gradyanları hesaplamak için kullanışlı bir yoldur.
