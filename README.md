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

**Feature Engineering**
ცვლადების ინჟინერიის პროცესში შეიქმნა Age ცვლადი, რომელიც სახლის ასაკს აღნიშნავს. ის არის 2025(მიმდინარე წლისა) და YearBuilt ცვლადის სხვაობა. შემდგომ YearBuilt ცვლადი წაიშალა.
ასევე დამატებით შეიქმნა სამი ცვლადი: total_area_1st_2nd_floor, total_area_1st_2nd_floor_bsmt,bsmt_diff, რომლებიც, შესაბამისად, 1-ლი და მე-2 სართულების, 1-ლი, მე-2 სართულებისა და სარდაფის და სრული და დაუსრულებელი სარდაფის ფართობების სხვაობას აღნიშნავს. სამიზნე ცვლადში არ არის აღმოჩენილი ანომალიური მნიშვნელობები(მაგალითად, უარყოფითი ფასი) და გამოტოვებული მნიშვნელობები.
დამატებით გადაიყარა ისეთი ცვლადები, რომლებშიც გამოტოვებული მნიშვნელობების წილი მეტი იყო 80%-ზე.
ასევე აღმოჩენილი არ იყო ისეთი ცვლადები, რომლებშიც უნიკალური მნიშვნელობების რაოდენობა იყო 1-ის ტოლი(წინააღმდეგ შემთხვევაში, ასეთი ცვლადი გადაიყრებოდა).
ვინაიდან ჩვენ პროექტში გამოვიყენეთ წრფივი მოდელი, საჭირო გახდა, გამოტოვებული მნიშვნელობების შევსება და კატეგორიული ცვლადების რიცხობრივად გარდაქმნა. ეს მიდგომები არ იქნა გათვალისწინებული LightGBM-ში, რადგან მას თავად შეუძლია გამკლავება კატეგორიულ ცვლადებთან და გამოტოვებულ მნიშვნელობებთან. 
წრივი მოდელის შემთხვევაში გამოტოვებული მნიშვნელობები 0-ებით შეივსო, ხოლო კატეგორიული ცვლადების შემთხვევაში იშვიათი კატეგორიები(ტოპ მე-3დან დაწყებული) გადაკეთდა კატეგორიად "სხვა" და მოხდა ცვლადების One Hot Encoding. კატეგორიების რაოდენობის ასეთი შემცირება გამოწვეული იყო იქედან, რომ მონაცემების ზომა დიდი არაა.


**Feature Selection**
ცვლადების შერჩევის ნაწილში გადაიყარა ისეთი ცვლადები, რომელთაც სამიზნე ცვლადთან 10% ან ნაკლები კორელაცია ჰქონდა. საბოლოო ეტაპზე, როდესაც LightGBM გაიწვრთნა, დადგენილი იქნა ისეთი ცვლადები, რომელთა მნიშვნელოვნება 0 იყო. შემდგომ ეს ცვლადები გადაიყარა და დარჩენილი ცვლადებით გაიწვრთნა ისევ LightGBM.

**Training**
პირველადი მოდელის გასაწვრთნელად გამოყენებული იყო წრფივი რეგრესია. მოდელის სიზუსტის შესაფასებლად თავდაპირველი მონაცემები(train.csv-ში არსებული), გაიყო ორ ნაწილად: სატრენინგო(80% მონაცემები) და სატესტო(20% მონაცემები). სიზუსტის შესაფასებლად შეირჩა შემდეგი მეტრიკები: MAPE, R2, RMSE. ინტუიციურობის გამო ძირითად მეტრიკად შეირჩა MAPE, ხოლო დანარჩენი ორი დამხმარე მეტრიკებად იყო არჩეული. overfitting-ის გასაკონტროლებლად შეირჩა სხვაობა სატესტო და სატრენინგო სიმრავლეების MAPE-ებს შორის. თუ ის 5%-ზე მეტია, მოდელის overfitting ჩაითვლებოდა დასაშვებზე მეტად და კომპლექსურობის შესამცირებლად ჰიპერპარამეტრების ცვლილება განიხილებოდა.
წრფივი რეგრესიის საბაზისო მოდელში(არა რეგულარიზებული) MAPE სატესტო სიმრავლეზე იყო 12.54% და სატრენინგოზე 10.95%. ამ შემთხვევაში overfitting-ის დონე იყო მცირე და სატესტო სიმრავლეზე არსებული 12.54% MAPE იყო გასაუმჯობესებელი.
შემდგომ Ridge რეგრესია გაიტესტა, რომელმაც ცოტა უფრო შეამცირა MAPE(overfitting აქაც არ გაზრდილა). საუკეთესო მეტრიკა სატესტო მონაცემებზე იყო 11.33%, რომელიც განსხვავებული რეგულარიზაციის პარამეტრის(lamdba) მნიშვნელობებით გადაირჩა.
უფრო კომპლექსურ ალგორითმად შეირჩა LightGBM, რომელიც ცხრილურ მონაცემებზე საკმაოდ დიდი სიზუსტით გამოირჩევა და აქვს უნარი გაუმკლავდეს კატეგორიულ და გამოტოვებულ მნიშვნელობებს თავისით.
თავდაპირველ LightGBM-ის მოდელში MAPE იყო 11.8%, თუმცა overfitting-ის დონე ოდნავ მაღალი იყო(მაგრამ არა საგანგაშო). bias-ის შესამცირებლად num_leaves, learning_rate და n_estimators პარამეტრების გაზრდა იყო საჭირო. ერთ-ერთი ასეთი გაშვების დროს overfitting დაახლოებით 5%-მდე ავიდა, თუმცა ასევე შემცირდა MAPE-ც(9.34%).
overfitting-ის შესამცირებლად გაიზარდა min_child_samples პარამეტრი, შემცირდა num_leaves პარამეტრი და ოდნავ შემცირდა num_estimators პარამეტრი.
LightGBM-ის დატრენინგების შემდეგ შეირჩა 0 მნიშვნელობების ცვლადები და ის გადაიყარა. საბოლოოდ კი დაახლოებით მსგავს პარამეტრებზე გაეშვა მოდელი, რომელშიც overfitting-ის დონე იყო მცირე(დაახლოებით 3%) და სატესტო მონაცემებზე MAPE იყო თავდაპირველ საბაზისო მოდელთან შემცირებული- 9.64%.

**MLflow Tracking**
აღნიშნული ტრენინგი ექსპერიმენტი დაილოგა mlflow-ს სერვერზე. ექსპერიმენტის სახელია house_price_prediction_regression. inference-ის ფაზაში საუკეთესო შედეგის მქონდე მოდელი წამოღებული იყოს პირდაპირ mlflow-ს სერვერიდან. mlflow-ზე დალოგილია მოდელის ჰიპერპარამეტრები, სატესტო/სატრენინგო სიმრავლეზე ნახსენები რეგრესიის მეტრიკები: MAPE, R2, RMSE. 
MAPE - საშუალო აბსოლუტური პროცენტული ცდომილება ნაპროგნოზებსა და რეალურ მნიშვნელობას შორის(რეალურ რიცხვითი მნიშვნელობასთან მიმართებით)
R2 - 1-ს გამოკლებული შეფარდება ნაპროგნოზები მნიშნელობების გაფანტულობისა და საშუალო პროგნოზით მიღებულ გაფანტულობას შორის(რაც მაღალია, მით უკეთესი).
RMSE - კვადრატული ფესვი საშუალო კვადრატული ცდომილებიდან.
საუკეთესო მოდელის run - nervous-stoat-922,  სატესტო MAPE 9.64%,  სატრენინგო MAPE 6.61%, ხოლო KAGGLE-ზე არსებული SCORE: 0.14423
ექსპერიმენტის ბმული: https://dagshub.com/nipkha21/House-Prices---Advanced-Regression-Techniques.mlflow/#/experiments/2?searchFilter=&orderByKey=attributes.start_time&orderByAsc=false&startTime=ALL&lifecycleFilter=Active&modelVersionFilter=All+Runs&datasetsFilter=W10%3D
