FROM registry.access.redhat.com/ubi8/python-39  

USER 0
WORKDIR /app

# 1️⃣ Instalar unzip y limpiar caches como root
RUN yum install -y unzip \
 && yum clean all

# 2️⃣ Copiar requisitos y fuentes
COPY requirements.txt .
COPY . .

# 3️⃣ Instalar dependencias Python
RUN pip install -U "pip>=19.3.1" \
 && pip3 install -r requirements.txt

# 4️⃣ Descomprimir modelo ZIP y eliminarlo
RUN unzip modelo_randomforest.zip -d . \
 && rm modelo_randomforest.zip

# Publicar archivos como no root
RUN chown -R 1001:0 /app
USER 1001

ENV FLASK_APP=app  
EXPOSE 5000  

CMD python app.py runserver 0.0.0.0:5000