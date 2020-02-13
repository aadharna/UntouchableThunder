module purge
module load jdk/11.0.4

echo `pwd`

HOME=`pwd`
cd /home/ad5238/.gradle
rm -rf caches/
rm -rf daemon/
rm -rf native/
rm -rf wrapper/
cd /home/ad5238/
rm -rf .gradle
cd /home/ad5238/UntouchableThunder/ext/GVGAI_GYM
./gradlew clean build --parallel

find ~/.gradle -type f -name "*.lock" -delete
echo `pwd`

cd $HOME
