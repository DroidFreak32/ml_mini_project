K
A
N
G

## Dependencies to install
```
numpy
pandas
sklearn
mysql database (wamp/xampp/lampp recommended)
```

Arch:

```
pacman -S python-numpy python-pandas python-sklearn
```
To install xampp in arch, use an AUR helper to install [xampp from here](https://aur.archlinux.org/packages/xampp)

Ubuntu:
```
apt install python3-numpy python3-pandas python3-sklearn
```

To import the database containing the dataset, first open phpmyadmin & create a database "iris_dataset".  
Select the database and go to the [import](http://127.0.0.1/phpmyadmin/db_import.php?db=iris_dataset) tab.  
Now select the db.sql file and import the dataset.  

Output:

![Output](https://raw.githubusercontent.com/DroidFreak32/ml_mini_project/master/2018-10-10_20%3A17%3A30.png)
