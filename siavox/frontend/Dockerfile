FROM nginx:alpine

# Remove default nginx config
RUN rm /etc/nginx/conf.d/default.conf

# Copy custom nginx configuration
COPY nginx.conf /etc/nginx/conf.d/default.conf

# Copy frontend source files to nginx server path
COPY index.html /usr/share/nginx/html/
COPY index.css /usr/share/nginx/html/
COPY app.js /usr/share/nginx/html/
COPY wav_encoder.js /usr/share/nginx/html/

EXPOSE 80

CMD ["nginx", "-g", "daemon off;"]
