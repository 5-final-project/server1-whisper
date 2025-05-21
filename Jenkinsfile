pipeline {
  agent { label 'team5' }

  environment {
    IMAGE_NAME = "whisper-stt-server"
    IMAGE_TAG = "${env.BUILD_NUMBER}"
  }

  stages {
    stage('Checkout') {
      steps {
        checkout scm
      }
    }
    stage('Build Docker Image') {
      steps {
        sh "docker build -t ${IMAGE_NAME}:${IMAGE_TAG} ."
      }
    }
    stage('Deploy Container') {
      steps {
        sh """
          if docker ps -a --format "{{.Names}}" | grep -q "^${IMAGE_NAME}\$"; then
            docker stop ${IMAGE_NAME}
            docker rm ${IMAGE_NAME}
          fi
          docker run -d --name ${IMAGE_NAME} -p 8001:8666 -v /mnt/d/team5/server1-whisper:/app/data ${IMAGE_NAME}:${IMAGE_TAG}
        """
      }
    }
    stage('Cleanup') {
      steps {
        script {
          // Remove dangling images
          sh "docker image prune -f"
        }
      }
    }
  }

  post {
    always {
      echo "Build #${env.BUILD_NUMBER} finished at ${new Date()}"
    }
  }
}