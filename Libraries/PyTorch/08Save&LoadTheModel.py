import torch
import torchvision.models as models

# Model Ağırlıklarını Kaydetme ve Yükleme
# PyTorch modelleri öğrenilen parametreleri state_dict adı verilen dahili bir durum sözlüğünde saklar. Bunlar torch.save yöntemi ile kalıcı hale getirilebilir:
model = models.vgg16(weights='IMAGENET1K_V1')
torch.save(model.state_dict(), f="C:/Users/firat/OneDrive/Belgeler/Projects/MLFundamentals/Downloaded/model_weights")

# Model ağırlıklarını yüklemek için, önce aynı modelin bir örneğini oluşturmanız ve ardından load_state_dict() yöntemini kullanarak parametreleri yüklemeniz gerekir.
# Aşağıdaki kodda, unpickling sırasında çalıştırılan işlevleri yalnızca ağırlıkların yüklenmesi için gerekli olanlarla sınırlamak için weights_only=True olarak ayarladık.
# Ağırlıklar yüklenirken weights_only=True kullanılması en iyi uygulama olarak kabul edilir.

model = models.vgg16()  # we do not specify ``weights``, i.e. create untrained model
model.load_state_dict(torch.load(f="C:/Users/firat/OneDrive/Belgeler/Projects/MLFundamentals/Downloaded/model_weights", weights_only=True))
model.eval()
# Not: Çıkarım yapmadan önce model.eval() yöntemini çağırarak dropout ve batch normalization katmanlarını değerlendirme moduna ayarladığınızdan emin olun. Bunu yapmamak tutarsız çıkarım sonuçları verecektir.

# Şekiller ile Modelleri Kaydetme ve Yükleme
# Model ağırlıklarını yüklerken, önce model sınıfını örneklememiz gerekir, çünkü bu sınıf bir ağın yapısını tanımlar.
# Bu sınıfın yapısını modelle birlikte kaydetmek isteyebiliriz, bu durumda kaydetme fonksiyonuna modeli aktarabiliriz:
torch.save(model, f="C:/Users/firat/OneDrive/Belgeler/Projects/MLFundamentals/Downloaded/saved_model")

# Daha sonra modeli aşağıda gösterildiği gibi yükleyebiliriz.
# torch.nn.Modules kaydetme ve yükleme bölümünde açıklandığı gibi, state_dict kaydetmek en iyi uygulama olarak kabul edilir.
# Ancak, aşağıda weights_only=False kullanıyoruz çünkü bu, torch.save için eski bir kullanım durumu olan modelin yüklenmesini içerir.
model = torch.load(f="C:/Users/firat/OneDrive/Belgeler/Projects/MLFundamentals/Downloaded/saved_model", weights_only=False),
# Not: Bu yaklaşım, modeli serileştirirken Python pickle modülünü kullanır, bu nedenle modeli yüklerken gerçek sınıf tanımının mevcut olmasına dayanır.
