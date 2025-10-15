<template>
    <div class="layOutContainer">
        <div class="title">
            <Title></Title>
        </div>
        <div class="main">
            <Main></Main>
        </div>
        <!-- <div class="footer">
            <Footer></Footer>
        </div> -->
    </div>
</template>

<script setup>
import Title from '../Layout/title/index.vue'
import Main from '../Layout/main/index.vue'
import Footer from '../Layout/footer/index.vue'

</script>

<style lang="scss" scoped>
.layOutContainer{
    width: 100%;
    height: 100vh;
    background-color: #f3f4f6;
    .title{
        width: 100%;
        height: 80px;
        background-color: #ffffff;
        position: fixed;
        top: 0;
        z-index: 1000;
    }
    .main{
        width: 100%;
        height: 100%;
        background-color: #f7f7f7;
    }
    .footer{
        width: 100%;
        height: 470px;
        background-color: #f7f7f7;
    }
}

</style>
