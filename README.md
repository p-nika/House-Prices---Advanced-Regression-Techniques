# House-Prices---Advanced-Regression-Techniques


პროექტის მიზანია აშშ-ს შტატ აიოვას ქალაქ ეიმსში სახლის ფასის პროგნოზირება სახლის მრავალფეროვანი პარამეტრების ხარჯზე.  აღნიშნული პროექტი წარმოადგენს რეგრესიის ამოცანას და ჩვენი მიზანია, მანქანური სწავლების ალგორითმებით შევქმნათ მოდელი, რომელიც კარგად იწინასწარმეტყველებს სახლის ფასს. ამ პროექტში გამოვიყენებთ წრფივ რეგრესიას(კლასიკურსაც და რეგულიზებულს) და უფრო კომპლექსურ ალგორითმს ცხრილური მონაცემებისთვის - LightGBM.

data საქაღალდე - სატრენინგო და სატესტო მონაცემების ფაილები
cat_columns.pkl - საბოლოო მოდელისთვის არსებული კატეგორიული ცვლადების სია, რომელთაც კატეგორიის მონაცემთა ტიპად გარდავქმნით(object-დან)
data_description.txt - მონაცემების აღწერა
model_experiment.ipynb - ამ ფაილში წარმოდგენილია მონაცემების ცვლადების შერჩევის, გაწმენდის, ცვლადების ინჟინერიისა და მოდელების დატრენინგების(მათ შორის MLFlow-ზე დალოგვის) ნაწილი
model_inference.ipynb - ამ ფაილში წარმოდგენილია კოდი, რომლითაც ჩვენ ვაკეთებთ პროგნოზს სატესტო მონაცემებზე
submission.csv - სატესტო მონაცემებზე არსებული პროგნოზი
train_columns.pkl - სატრენინგო სიმრავლეში არსებული ცვლადების სახელები
utils.py - მონაცემების პრეპროცესინგის საჭირო ფუნქცია(რათა სატესტო მონაცემები დავიყვანოთ სატრენინგოს სტრუქტურამდე)

**FEATURE ENGINEERING**
