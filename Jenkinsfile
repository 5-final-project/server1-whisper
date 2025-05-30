pipeline {
  agent { label 'team5' }

  environment {
    IMAGE_NAME = "server2-rag-pipeline"
    IMAGE_TAG  = "${env.BUILD_NUMBER}"
    GEMINI_API_KEY = credentials('GEMINI_API_KEY')
    VECTOR_API_URL  = credentials('VECTOR_API_URL')
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
        sh '''
          # legacy container 제거 (이름이 agentic_rag 라면)
          if docker ps -a --filter "name=^/agentic_rag$" --format "{{.Names}}" | grep -q "^agentic_rag$"; then
            echo "Removing legacy container agentic_rag"
            docker rm -f agentic_rag
          fi
          
          # server2-rag 컨테이너 제거 (새 이름)
          if docker ps -a --filter "name=^/server2-rag$" --format "{{.Names}}" | grep -q "^server2-rag$"; then
            echo "Removing existing container server2-rag"
            docker rm -f server2-rag
          fi

          # 새 컨테이너 실행 (Team5 전용 설정)
          docker run -d \
            --name server2-rag \
            --network team5-net \
            -p 8125:8125 \
            -e GEMINI_API_KEY="${GEMINI_API_KEY}" \
            -e VECTOR_API_URL="${VECTOR_API_URL}" \
            -e ENABLE_METRICS=true \
            -e SERVICE_NAME=server2-rag \
            -v /var/logs/server2_rag:/var/logs/server2_rag \
            ${IMAGE_NAME}:${IMAGE_TAG}
        '''
      }
    }
    stage('Health Check') {
      steps {
        sh '''
          # 컨테이너가 정상적으로 시작될 때까지 대기
          echo "Waiting for container to start..."
          sleep 30
          
          # 컨테이너 상태 확인
          if docker ps --filter "name=server2-rag" --filter "status=running" | grep -q server2-rag; then
            echo "Container is running"
          else
            echo "Container failed to start"
            docker logs server2-rag
            exit 1
          fi
          
          # 헬스체크
          for i in {1..10}; do
            if curl -f http://localhost:8125/ 2>/dev/null; then
              echo "Health check passed"
              break
            else
              echo "Health check attempt $i failed, retrying..."
              sleep 5
            fi
            if [ $i -eq 10 ]; then
              echo "Health check failed after 10 attempts"
              docker logs server2-rag
              exit 1
            fi
          done
          
          # 메트릭 엔드포인트 확인
          if curl -f http://localhost:8125/metrics 2>/dev/null; then
            echo "Metrics endpoint is working"
          else
            echo "Metrics endpoint check failed"
            exit 1
          fi
          
          echo "Server2-rag 컨테이너가 정상적으로 시작되었습니다."
        '''
      }
    }
    stage('Cleanup') {
      steps {
        sh '''
          # 이전 이미지들 정리 (최신 2개만 유지)
          docker images ${IMAGE_NAME} --format "table {{.Tag}}\t{{.ID}}" | tail -n +2 | sort -nr | tail -n +3 | awk '{print $2}' | xargs -r docker rmi
          # dangling 이미지 정리
          docker image prune -f
        '''
      }
    }
  }

  post {
    always {
      echo "Build #${env.BUILD_NUMBER} finished at ${new Date()}"
    }
    success {
      echo "Server2-rag 배포가 성공적으로 완료되었습니다."
      echo "서비스 확인: http://localhost:8125/"
      echo "메트릭 확인: http://localhost:8125/metrics"
    }
    failure {
      echo "Server2-rag 배포에 실패했습니다."
      sh '''
        echo "=== Container Status ==="
        docker ps -a --filter "name=server2-rag"
        echo ""
        echo "=== Container Logs ==="
        docker logs server2-rag 2>&1 || echo "No logs available"
        echo ""
        echo "=== Network Status ==="
        docker network ls | grep team5 || echo "team5-net not found"
      '''
    }
  }
}