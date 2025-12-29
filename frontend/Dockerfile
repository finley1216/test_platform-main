FROM node:18-alpine

WORKDIR /app

# Copy package files
COPY package*.json ./

# Install dependencies
RUN npm cache clean --force && \
    npm install

# Copy frontend source
COPY . .

# 不設置 REACT_APP_API_BASE，讓前端根據 hostname 自動選擇
# ENV REACT_APP_API_BASE=$REACT_APP_API_BASE  # 已移除，改為自動檢測

# Expose React dev server port
EXPOSE 3000

# Start the development server
CMD ["npm", "start"]

