FROM node:18-alpine

WORKDIR /app

# Copy package files
COPY package*.json ./

# Install dependencies
RUN npm cache clean --force && \
    npm install

# Copy frontend source
COPY . .

# Set API base URL for Docker environment
ENV REACT_APP_API_BASE=$REACT_APP_API_BASE

# Expose React dev server port
EXPOSE 3000

# Start the development server
CMD ["npm", "start"]

