FROM nginx:alpine

# Copy the NGINX configuration file into the container
COPY nginx.conf /etc/nginx/conf.d/default.conf

CMD ["nginx", "-g", "daemon off;"]