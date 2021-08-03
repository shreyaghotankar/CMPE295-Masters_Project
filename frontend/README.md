## About React configurations

This project was bootstrapped with [Create React App](https://github.com/facebook/create-react-app).

## Testing

Jest library
`npm test`

## Branching


 > ! Before you start always pull latest changes \
 > `git pull`

For each feature/bug or issue resolvment create a new branch 

`git checkout -b [branch name]`

Follow branch naming convention:
[feature/bug/issue]/[name of the reature]-[author]
Ex: feature/readme-az


## Amplify Setup

```
npm install -g @aws-amplify/cli
amplify upgrade
```

AWS best practices suggest to create a new user for each project and grant granual permissions to it. 

After the user is create in AWS console and creadentials for CLI (access key Id and secret access key) are recieved, aws cli should be cofigured on a local machine. 

Run the command to create a new profile and you will be prompted to add credentials: 
```
aws configure --profile [new_profile_name]
```

Output: 
```
AWS Access Key ID [None]: SOMEKEYID
AWS Secret Access Key [None]: SoMeSeCrEtHeRe
Default region name [None]: us-west-2 // or any other region of your choice 
Default output format [None]: json // or any other format
```

To avoid specifying the profile in every command set the AWS_PROFILE environment variable: 
```
export AWS_PROFILE=master_project
```

## References:

[Configure AWS CLI](https://docs.aws.amazon.com/cli/latest/userguide/cli-chap-configure.html) 
