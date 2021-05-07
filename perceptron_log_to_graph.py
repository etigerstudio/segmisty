import seaborn as sns
import matplotlib.pyplot as plt
import re
sns.set_theme(style="darkgrid")
plt.figure(dpi=360)


with open("train_log.txt") as f:
    content = f.readlines()

t = re.compile(r"training.*F1:")
e = re.compile(r"evaluation.*F1:")
tf = []
ef = []
for l in content:
    tm = t.match(l)
    if tm is None:
        em = e.match(l)
        if em is not None:
            ef.append(float(l[em.span()[1]:em.span()[1]+6]))
    else:
        tf.append(float(l[tm.span()[1]:tm.span()[1]+6]))

ef=ef[0:-1]
tf = tf[0:-1:10]
data = {"epoch":range(0,50),"test_F1":ef,"training_F1":tf}
sns.lineplot(x="epoch",y="test_F1",data=data)
sns.lineplot(x="epoch",y="training_F1",data=data)
plt.ylim(0.5,1.05)
plt.xlim(-1,51)
plt.xticks(range(0,51,5),range(0, 501, 50))
plt.legend(["test.txt", "training.txt"])
plt.figure(dpi=1200)
plt.show()