# 将params模型放到该目录下
mv ../bin /usr/local/bin/detection-server
cp detection-server /etc/init.d/
chkconfig --add detection-server
chkconfig detection-server on
