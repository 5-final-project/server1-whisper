pipeline {
  agent any

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
        script {
          def built = docker.build("${IMAGE_NAME}:${IMAGE_TAG}")
        }
      }
    }
    stage('Deploy Container') {
      steps {
        script {
          if (docker.image("${IMAGE_NAME}:${IMAGE_TAG}").exists()) {
            sh """
              if docker ps -a --format "{{.Names}}" | grep -q "^${IMAGE_NAME}\\\\\$"; then
                docker stop ${IMAGE_NAME} && docker rm ${IMAGE_NAME}
              fi
            """
          }
          docker.image("${IMAGE_NAME}:${IMAGE_TAG}").run("-d --name ${IMAGE_NAME} -p 8666:8666 -v /mnt/d/team5/server1-whisper:/app/data")
        }
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