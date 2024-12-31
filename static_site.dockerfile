FROM node as build

# Allow for compose files to configure the final domain
ARG QUERY_URL=/api
ENV REACT_APP_QUERY_BASE=$QUERY_URL

# Copy webapp
COPY static_site static_site
WORKDIR static_site/transferware-app

# Download deps
RUN npm install

# Build app
RUN npm run build

FROM nginx
COPY --from=build /static_site/transferware-app/build /usr/share/nginx/html
EXPOSE 8080