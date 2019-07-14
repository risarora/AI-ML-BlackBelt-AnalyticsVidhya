## Tensor Flow Serving
tYWwEN4YzR3G


cd /tmp
curl -LO 'https://tensorflow.org/images/blogs/serving/cat.jpg'


Send request to the server
```
resnet_client_cc --image_file=cat.jpg

```

```
serving/ ***
  ***                 https://docs.bitnami.com/google/ ***
  *** Bitnami Forums: https://community.bitnami.com/ ***
rishiarora_in@tensorflowserving-1-vm:~$ cd /tmp
rishiarora_in@tensorflowserving-1-vm:/tmp$ curl -LO 'https://tensorflow.org/images/blogs/serving/cat.jpg'
  % Total    % Received % Xferd  Average Speed   Time    Time     Time  Current
                                 Dload  Upload   Total   Spent    Left  Speed
100   252  100   252    0     0  11788      0 --:--:-- --:--:-- --:--:-- 12000
100 63753  100 63753    0     0   377k      0 --:--:-- --:--:-- --:--:--  377k
rishiarora_in@tensorflowserving-1-vm:/tmp$ resnet_client_cc --image_file=cat.jpg
calling predict using file: cat.jpg  ...
call predict ok
outputs size is 2
the result tensor[0] is:
[2.41628254e-06 1.90121955e-06 2.72477027e-05 4.4263885e-07 8.98362089e-07 6.84422412e-06 1.66555201e-05 3.4298439e-06 5.25692e-06 2.66782135e-05...]...
the result tensor[1] is:
286
Done.
rishiarora_in@tensorflowserving-1-vm:/tmp$ ifconfig
-bash: ifconfig: command not found
rishiarora_in@tensorflowserving-1-vm:/tmp$ ipconfig
-bash: ipconfig: command not found
rishiarora_in@tensorflowserving-1-vm:/tmp$ ping google.com
PING google.com (172.217.212.101) 56(84) bytes of data.
64 bytes from 172.217.212.101: icmp_seq=1 ttl=50 time=5.50 ms
64 bytes from 172.217.212.101: icmp_seq=2 ttl=50 time=0.368 ms
64 bytes from 172.217.212.101: icmp_seq=3 ttl=50 time=0.398 ms
64 bytes from 172.217.212.101: icmp_seq=4 ttl=50 time=0.382 ms
^C
--- google.com ping statistics ---
4 packets transmitted, 4 received, 0% packet loss, time 3048ms
rtt min/avg/max/mdev = 0.368/1.664/5.508/2.219 ms
rishiarora_in@tensorflowserving-1-vm:/tmp$ resnet_client_cc --image_file=cat.jpg
calling predict using file: cat.jpg  ...
call predict ok
outputs size is 2
the result tensor[0] is:
286
the result tensor[1] is:
[2.41628254e-06 1.90121955e-06 2.72477027e-05 4.4263885e-07 8.98362089e-07 6.84422412e-06 1.66555201e-05 3.4298439e-06 5.25692e-06 2.66782135e-05...]...
Done.
rishiarora_in@tensorflowserving-1-vm:/tmp$ cat /opt/bitnami/tensorflow-serving/conf/tensorflow-serving.conf
model_config_list: {
config: {
name: "resnet",
base_path: "/opt/bitnami/model-data",
model_platform: "tensorflow",
}
}
rishiarora_in@tensorflowserving-1-vm:/tmp$ ls /opt/bitnami/modal-data
ls: cannot access '/opt/bitnami/modal-data': No such file or directory
rishiarora_in@tensorflowserving-1-vm:/tmp$ ls /opt/bitnami/model-data
1538687457
rishiarora_in@tensorflowserving-1-vm:/tmp$ ls /opt/bitnami/model-data/1538687457/
saved_model.pb  variables
rishiarora_in@tensorflowserving-1-vm:/tmp$ ls
cat.jpg              nami_1562871149.log
nami_1562871136.log  nami_1562871151.log
nami_1562871140.log  nami_1562871196.log
nami_1562871142.log  resnet_v2_fp32_savedmodel_NHWC_jpg.tar.gz
nami_1562871144.log  ssh-7203lmvzP6
nami_1562871146.log  systemd-private-70f53d957db24bce874236ba858aab7e-haveged.service-Tlejnz
nami_1562871148.log
rishiarora_in@tensorflowserving-1-vm:/tmp$ ls /opt/bitnami/model-data/1538687457/
saved_model.pb  variables
```
  
