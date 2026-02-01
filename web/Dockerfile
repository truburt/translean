# Copyright © 2026 Vladimir Vaulin-Belskii. All rights reserved.
# Builder stage
FROM node:20-slim AS builder
WORKDIR /app
COPY web/package.json web/package-lock.json ./
RUN npm ci
COPY web/ .
RUN npm run build

# Runner stage
FROM nginx:stable-alpine
COPY ./web/nginx.conf /etc/nginx/conf.d/default.conf
COPY --from=builder /app/dist /usr/share/nginx/html

